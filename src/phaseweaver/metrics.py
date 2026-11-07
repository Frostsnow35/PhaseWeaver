"""Metrics replayed from the event journal (ADR-0003: no parallel counters)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any

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
    #: seconds during which at least one unit was decoding/prefilling
    gpu_busy_s: float = 0.0
    #: integral of the number of concurrent reasoning units over time
    concurrency_s: float = 0.0
    #: tasks that were ready but waiting for capacity, sampled per event
    queue_events: int = 0
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
    def gpu_busy_fraction(self) -> float:
        """How much of the wall clock the device was doing inference.

        A low value means the run is dominated by tool phases (inherent) rather
        than by scheduler inefficiency - this is what explains a large
        optimality gap without hand-waving.
        """
        return (self.gpu_busy_s / self.wall_clock_s) if self.wall_clock_s > 0 else 0.0

    @property
    def mean_concurrency(self) -> float:
        """Average number of streams sharing the device.

        This is the number the α throughput model cares about: a run can be
        "busy" 99% of the time while averaging 1.5 streams, which is exactly why
        a bound computed for k_max streams looks unreachable.
        """
        return (self.concurrency_s / self.wall_clock_s) if self.wall_clock_s > 0 else 0.0

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
            "gpu_busy": round(self.gpu_busy_fraction, 3),
        }


class _ReplayState:
    """Incremental fold over the event log.

    Split out of `MetricsReducer.replay` because one 70-line function with a dozen
    branches is where metric bugs hide. Each event type now has one small handler
    and the dispatcher is a dict lookup.
    """

    def __init__(self, run_id: str) -> None:
        self.report = RunReport(run_id=run_id)
        self._arrivals: dict[str, tuple[float, float]] = {}
        self._resident: dict[str, int] = {}
        self._reasoning: set[str] = set()
        self._last_ts = 0.0
        self._idle = 0.0
        self._waste = 0.0
        self._streams = 0
        self._handlers: dict[str, Callable[[Any], None]] = {
            "unit_created": self._on_unit_created,
            "phase_transitioned": self._on_phase,
            "engine_active": self._on_engine_active,
            "context_state_changed": self._on_context,
            "park_completed": self._on_park,
            "thrash": self._on_event_count("thrash_count"),
            "admission_rejected": self._on_event_count("admission_rejects"),
            "restore_completed": self._on_restore,
        }

    # --- time integration -------------------------------------------------
    def step(self, event: Any) -> None:
        dt = max(0.0, event.ts - self._last_ts)
        if self._reasoning:
            self.report.gpu_busy_s += dt
        self.report.concurrency_s += dt * self._streams
        resident_total = sum(self._resident.values())
        if not self._reasoning:
            self._idle += dt * resident_total
        for unit_id, size in self._resident.items():
            if unit_id not in self._reasoning and size > 0:
                self._waste += dt * size
        self._last_ts = event.ts
        handler = self._handlers.get(event.event)
        if handler is not None:
            handler(event)

    # --- handlers ---------------------------------------------------------
    def _on_unit_created(self, event: Any) -> None:
        payload = event.payload
        if event.unit_id is None:
            return
        if payload.get("kind") not in ("agent", "subagent"):
            return
        if "arrival_ts" not in payload:
            return
        self._arrivals[event.unit_id] = (
            float(payload["arrival_ts"]),
            float(payload.get("deadline") or 0.0),
        )

    def _on_phase(self, event: Any) -> None:
        if event.unit_id is None:
            return
        to = str(event.payload.get("to", ""))
        if to == "REASONING":
            self._reasoning.add(event.unit_id)
        elif to in ("ACTING", "WAITING", "COMPLETED", "FAILED", "CANCELLED"):
            self._reasoning.discard(event.unit_id)
        if to in ("COMPLETED", "FAILED", "CANCELLED"):
            self._finish_task(event, to)

    def _finish_task(self, event: Any, outcome: str) -> None:
        if event.unit_id not in self._arrivals:
            return
        arrived, deadline = self._arrivals.pop(event.unit_id)
        latency_ms = (event.ts - arrived) * 1000
        self.report.latencies_ms.append(latency_ms)
        # deadline is absolute; compare the finish time against it, otherwise a
        # duration would always appear to meet it.
        hit = (event.ts <= deadline) if deadline > 0 else True
        if outcome == "COMPLETED":
            self.report.completed += 1
            if hit:
                self.report.slo_hits += 1
        elif outcome == "FAILED":
            self.report.failed += 1
        else:
            self.report.cancelled += 1

    def _on_engine_active(self, event: Any) -> None:
        delta = int(event.payload.get("delta", 0))
        self._streams = max(0, self._streams + delta)

    def _on_context(self, event: Any) -> None:
        if event.unit_id is None:
            return
        self._resident[event.unit_id] = int(event.payload.get("resident_bytes", 0))
        self.report.peak_resident_bytes = max(
            self.report.peak_resident_bytes, sum(self._resident.values())
        )

    def _on_park(self, event: Any) -> None:
        self.report.park_count += 1

    def _on_restore(self, event: Any) -> None:
        self.report.restore_latencies_ms.append(float(event.payload.get("latency_ms", 0.0)))
        self.report.recompute_tokens += int(event.payload.get("recompute_tokens", 0))

    def _on_event_count(self, attribute: str) -> Callable[[Any], None]:
        def handler(_event: Any) -> None:
            setattr(self.report, attribute, getattr(self.report, attribute) + 1)

        return handler

    def finish(self) -> RunReport:
        self.report.wall_clock_s = self._last_ts
        self.report.gpu_idle_wasted_bytes_s = self._idle
        self.report.kv_occupancy_waste_bytes_s = self._waste
        return self.report


class MetricsReducer:
    @staticmethod
    def replay(journal: EventJournal, *, clock: Clock | None = None) -> RunReport:
        state = _ReplayState(journal.run_id)
        for event in journal.events():
            state.step(event)
        return state.finish()
