"""Metrics replayed from the event journal (ADR-0003: no parallel counters)."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

from phaseweaver.clock import Clock
from phaseweaver.journal import EventJournal


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


@dataclass
class RunReport:
    run_id: str
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    slo_hits: int = 0
    wall_clock_s: float = 0.0
    park_count: int = 0
    thrash_count: int = 0
    recompute_tokens: int = 0
    peak_resident_bytes: int = 0
    gpu_idle_wasted_bytes_s: float = 0.0
    #: F4: bytes x seconds where a context occupies memory while its owner is
    #: NOT reasoning. This is the "pinned while idle" waste, and unlike
    #: gpu_idle_wasted_bytes_s it is non-zero whenever anyone yields badly.
    kv_occupancy_waste_bytes_s: float = 0.0
    admission_rejects: int = 0
    restore_latencies_ms: list[float] = field(default_factory=list)
    queue_waits_ms: list[float] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.completed + self.failed + self.cancelled

    @property
    def slo_attainment_rate(self) -> float:
        return (self.slo_hits / self.total) if self.total else 0.0

    @property
    def goodput(self) -> float:
        return (self.slo_hits / self.wall_clock_s) if self.wall_clock_s > 0 else 0.0

    @property
    def latency_p50_ms(self) -> float:
        return percentile(self.latencies_ms, 50)

    @property
    def latency_p95_ms(self) -> float:
        return percentile(self.latencies_ms, 95)

    @property
    def latency_p99_ms(self) -> float:
        return percentile(self.latencies_ms, 99)

    @property
    def latency_cv(self) -> float:
        if len(self.latencies_ms) < 2:
            return 0.0
        m = mean(self.latencies_ms)
        return (pstdev(self.latencies_ms) / m) if m else 0.0

    @property
    def thrash_rate(self) -> float:
        minutes = self.wall_clock_s / 60.0
        return (self.thrash_count / minutes) if minutes > 0 else 0.0

    @property
    def jain_fairness(self) -> float:
        """Jain index over task latencies: 1.0 = perfectly equal, 1/n = one hog."""
        if len(self.latencies_ms) < 2:
            return 1.0
        total = sum(self.latencies_ms)
        sq = sum(v * v for v in self.latencies_ms)
        return (total * total) / (len(self.latencies_ms) * sq) if sq else 1.0

    def summary_row(self) -> dict[str, float]:
        return {
            "completed": float(self.completed),
            "slo_rate": round(self.slo_attainment_rate, 4),
            "goodput": round(self.goodput, 4),
            "p50_ms": round(self.latency_p50_ms, 1),
            "p95_ms": round(self.latency_p95_ms, 1),
            "p99_ms": round(self.latency_p99_ms, 1),
            "lat_cv": round(self.latency_cv, 3),
            "thrash": round(self.thrash_rate, 2),
            "recompute_tok": float(self.recompute_tokens),
            "peak_kv_mib": round(self.peak_resident_bytes / 1024 / 1024, 1),
            "idle_waste_mib_s": round(self.gpu_idle_wasted_bytes_s / 1024 / 1024, 1),
            "kv_waste_mib_s": round(self.kv_occupancy_waste_bytes_s / 1024 / 1024, 1),
            "jain": round(self.jain_fairness, 3),
        }


class MetricsReducer:
    @staticmethod
    def replay(journal: EventJournal, *, clock: Clock | None = None) -> RunReport:
        report = RunReport(run_id=journal.run_id)
        arrivals: dict[str, tuple[float, float]] = {}
        resident: dict[str, int] = {}
        reasoning_units: set[str] = set()
        last_ts = 0.0
        idle_accum = 0.0
        waste_accum = 0.0

        def resident_total() -> int:
            return sum(resident.values())

        for ev in journal.events():
            dt = max(0.0, ev.ts - last_ts)
            if not reasoning_units:
                idle_accum += dt * resident_total()
            for uid, size in resident.items():
                if uid not in reasoning_units and size > 0:
                    waste_accum += dt * size
            last_ts = ev.ts

            if ev.event == "unit_created":
                payload = ev.payload
                is_task_unit = payload.get("kind") in ("agent", "subagent")
                if ev.unit_id and is_task_unit and "arrival_ts" in payload:
                    arrivals[ev.unit_id] = (
                        float(payload["arrival_ts"]),
                        float(payload.get("deadline") or 0.0),
                    )
            elif ev.event == "phase_transitioned" and ev.unit_id:
                to = str(ev.payload.get("to", ""))
                if to == "REASONING":
                    reasoning_units.add(ev.unit_id)
                elif to in ("ACTING", "WAITING", "COMPLETED", "FAILED", "CANCELLED"):
                    reasoning_units.discard(ev.unit_id)
                if to in ("COMPLETED", "FAILED", "CANCELLED") and ev.unit_id in arrivals:
                    arrived, deadline = arrivals.pop(ev.unit_id)
                    latency_ms = (ev.ts - arrived) * 1000
                    report.latencies_ms.append(latency_ms)
                    # deadline is an absolute clock value; compare the finish time
                    # against it (comparing a duration against it would always pass).
                    hit = (ev.ts <= deadline) if deadline > 0 else True
                    if to == "COMPLETED":
                        report.completed += 1
                        if hit:
                            report.slo_hits += 1
                    elif to == "FAILED":
                        report.failed += 1
                    else:
                        report.cancelled += 1
            elif ev.event == "context_state_changed" and ev.unit_id:
                resident[ev.unit_id] = int(ev.payload.get("resident_bytes", 0))
                report.peak_resident_bytes = max(report.peak_resident_bytes, resident_total())
            elif ev.event == "park_completed":
                report.park_count += 1
            elif ev.event == "thrash":
                report.thrash_count += 1
            elif ev.event == "admission_rejected":
                report.admission_rejects += 1
            elif ev.event == "restore_completed":
                report.restore_latencies_ms.append(float(ev.payload.get("latency_ms", 0.0)))
                report.recompute_tokens += int(ev.payload.get("recompute_tokens", 0))

        report.wall_clock_s = last_ts
        report.gpu_idle_wasted_bytes_s = idle_accum
        report.kv_occupancy_waste_bytes_s = waste_accum
        return report
