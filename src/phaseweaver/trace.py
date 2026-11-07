"""Per-unit post-mortem: where did this task spend its time, and why?

Answering "why was this task slow" from a 10k-line event log is the difference
between an observable runtime and a black box. The attribution is deliberately
coarse and honest:

- time in `REASONING`   -> the model/engine is the limiter;
- time in `ACTING`      -> the tool is the limiter;
- time in `QUEUED/WAITING` -> admission or capacity is the limiter;
- time parked          -> KV disposition decisions cost this unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from phaseweaver.journal import EventJournal
from phaseweaver.timeline import Timeline, build_timeline


@dataclass
class UnitTrace:
    unit_id: str
    latency_s: float
    phases: dict[str, float] = field(default_factory=dict)
    context_states: dict[str, float] = field(default_factory=dict)
    recompute_tokens: int = 0
    park_count: int = 0
    restore_count: int = 0
    handoffs: int = 0
    narrative: list[str] = field(default_factory=list)

    @property
    def dominant(self) -> tuple[str, float]:
        if not self.phases:
            return ("unknown", 0.0)
        name = max(self.phases, key=lambda key: self.phases[key])
        return (name, self.phases[name])

    def hint(self) -> str:
        name, seconds = self.dominant
        total = max(1e-9, sum(self.phases.values()))
        share = seconds / total
        if name in ("QUEUED", "WAITING"):
            return (
                f"{name} 占 {share:.0%}:准入/容量是瓶颈——检查预算与 max_inflight,"
                "或让出策略是否不够激进"
            )
        if name == "ACTING":
            return f"ACTING 占 {share:.0%}:工具本身是瓶颈——这不是调度能解决的"
        if name == "REASONING":
            return f"REASONING 占 {share:.0%}:模型/引擎是瓶颈——看上下文长度与解码速率"
        return f"{name} 占 {share:.0%}"


def trace_unit(journal: EventJournal, unit_id: str, timeline: Timeline | None = None) -> UnitTrace:
    tl = timeline or build_timeline(journal)
    phases: dict[str, float] = {}
    states: dict[str, float] = {}
    for span in tl.for_unit(unit_id):
        target = phases if span.kind == "phase" else states
        target[span.name] = target.get(span.name, 0.0) + span.duration

    trace = UnitTrace(
        unit_id=unit_id,
        latency_s=sum(phases.values()),
        phases={k: round(v, 3) for k, v in phases.items()},
        context_states={k: round(v, 3) for k, v in states.items()},
    )

    for event in journal.events():
        if event.unit_id != unit_id:
            continue
        payload = event.payload
        if event.event == "restore_completed":
            trace.restore_count += 1
            trace.recompute_tokens += int(payload.get("recompute_tokens", 0))
        elif event.event == "park_completed":
            trace.park_count += 1
            applied = payload.get("applied", "?")
            trace.narrative.append(
                f"t={event.ts:7.2f}  park({applied}) "
                f"latency={float(payload.get('latency_ms', 0)):.0f}ms"
            )
        elif event.event == "lease_transferred":
            trace.handoffs += 1
            direction = payload.get("direction", "?")
            trace.narrative.append(
                f"t={event.ts:7.2f}  kv-handoff({direction}) tokens={payload.get('tokens')}"
            )
        elif event.event == "phase_transitioned":
            to = payload.get("to", "?")
            reason = payload.get("reason")
            suffix = f" reason={reason}" if reason else ""
            trace.narrative.append(f"t={event.ts:7.2f}  -> {to}{suffix}")
        elif event.event == "tool_finished":
            trace.narrative.append(
                f"t={event.ts:7.2f}  tool {payload.get('tool')} "
                f"{payload.get('status')} {payload.get('duration_ms')}ms"
            )
    return trace


def format_trace(trace: UnitTrace) -> str:
    lines = [
        f"unit: {trace.unit_id}",
        f"latency: {trace.latency_s:.2f}s   recompute: {trace.recompute_tokens} tokens",
        f"parks: {trace.park_count}   restores: {trace.restore_count}   handoffs: {trace.handoffs}",
        "phase time: " + ", ".join(f"{k}={v:.2f}s" for k, v in sorted(trace.phases.items())),
        f"diagnosis: {trace.hint()}",
        "timeline:",
    ]
    lines.extend("  " + line for line in trace.narrative)
    return "\n".join(lines)
