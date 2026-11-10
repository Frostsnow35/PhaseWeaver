"""Replay a user's own agent trace through the deterministic simulator.

JSONL schema (one event per line, stdlib only)::

    {"ts": 0.0, "unit_id": "a0", "kind": "reason",
     "prompt_tokens": 4096, "out_tokens": 128,
     "prefix_id": "sys-v1", "prefix_tokens": 3072}
    {"ts": 3.2, "unit_id": "a0", "kind": "tool",
     "name": "pytest", "tool_s": 5.0}

``prefix_id`` / ``prefix_tokens`` are optional. Without them the prefix overlap
report is ``unknown`` — never a guessed number.

Evidence level: every number produced here is ``[L1]`` deterministic simulation,
not a real-GPU measurement. The prefix overlap is a static property of the trace
(block-level dedup potential), not a performance claim.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from phaseweaver.app import build_policy
from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.leases import Budget
from phaseweaver.metrics import MetricsReducer, RunReport
from phaseweaver.runtime import AgentFn, AgentHandle, Runtime, RuntimeConfig, TaskOutcome
from phaseweaver.tools.base import ToolSpec

BLOCK_TOKENS = 16
BYTES_PER_TOKEN = 57344  # Qwen2.5-7B bf16 KV per token
PREFILL_TOK_PER_S = 3200
DECODE_TOK_PER_S = 45


@dataclass(frozen=True)
class TraceEvent:
    ts: float
    unit_id: str
    kind: str  # "reason" | "tool"
    prompt_tokens: int = 0
    out_tokens: int = 0
    tool_s: float = 0.0
    name: str = "tool"
    prefix_id: str | None = None
    prefix_tokens: int = 0


@dataclass(frozen=True)
class PrefixOverlap:
    reason_events: int
    logical_blocks: int
    distinct_blocks: int
    saving_ratio: float | None

    @property
    def label(self) -> str:
        if self.saving_ratio is None:
            return "unknown (no prefix_id/prefix_tokens in trace)"
        return f"{self.saving_ratio * 100:.1f}%"


@dataclass(frozen=True)
class ReplayResult:
    evidence: str
    trace_events: int
    units: int
    capacity_gib: float
    seeds: tuple[int, ...]
    slo_s: float
    prefix_hit_ratio: float
    prefix: PrefixOverlap
    policies: dict[str, dict[str, float]]


def _required_str(data: dict[str, object], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: '{key}' must be a non-empty string")
    return value


def _number(data: dict[str, object], key: str, default: float, where: str) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: '{key}' must be a number")
    result = float(value)
    if result < 0:
        raise ValueError(f"{where}: '{key}' must be >= 0")
    return result


def _parse_event(data: dict[str, object], where: str) -> TraceEvent:
    kind = _required_str(data, "kind", where)
    if kind not in ("reason", "tool"):
        raise ValueError(f"{where}: kind must be 'reason' or 'tool'")
    prefix_id = data.get("prefix_id")
    if prefix_id is not None and not isinstance(prefix_id, str):
        raise ValueError(f"{where}: 'prefix_id' must be a string or null")
    name = data.get("name", "tool")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}: 'name' must be a non-empty string")
    return TraceEvent(
        ts=_number(data, "ts", 0.0, where),
        unit_id=_required_str(data, "unit_id", where),
        kind=kind,
        prompt_tokens=int(_number(data, "prompt_tokens", 0.0, where)),
        out_tokens=int(_number(data, "out_tokens", 0.0, where)),
        tool_s=_number(data, "tool_s", 0.0, where),
        name=name,
        prefix_id=prefix_id,
        prefix_tokens=int(_number(data, "prefix_tokens", 0.0, where)),
    )


def load_trace(path: Path) -> list[TraceEvent]:
    """Read a JSONL trace. Blank lines and ``#`` comments are ignored."""
    events: list[TraceEvent] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{path}:{lineno}"
        try:
            obj: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{where}: each line must be a JSON object")
        events.append(_parse_event(cast(dict[str, object], obj), where))
    if not events:
        raise ValueError(f"{path}: empty trace")
    return events


def prefix_overlap(events: list[TraceEvent], block_tokens: int = BLOCK_TOKENS) -> PrefixOverlap:
    """Static block-level dedup potential of the shared prefixes in a trace.

    ``logical_blocks`` counts a prefix once per reason event (what an engine
    without prefix sharing stores). ``distinct_blocks`` counts it once per
    ``prefix_id``. The difference is the upper bound on bytes a refcounted
    prefix cache could remove — not a measured speedup.
    """
    reasons = [e for e in events if e.kind == "reason"]
    tagged = [e for e in reasons if e.prefix_id is not None and e.prefix_tokens > 0]
    if not tagged:
        return PrefixOverlap(len(reasons), 0, 0, None)
    logical = sum((e.prefix_tokens + block_tokens - 1) // block_tokens for e in tagged)
    longest: dict[str, int] = {}
    for event in tagged:
        key = event.prefix_id or ""
        longest[key] = max(longest.get(key, 0), event.prefix_tokens)
    distinct = sum((n + block_tokens - 1) // block_tokens for n in longest.values())
    ratio = 1.0 - distinct / logical if logical else 0.0
    return PrefixOverlap(len(reasons), logical, distinct, ratio)


def estimate_slo(events: list[TraceEvent]) -> float:
    """A deadline derived from the slowest unit's own trace, not a magic number."""
    per_unit: dict[str, float] = {}
    for event in events:
        work = (
            event.tool_s
            + event.prompt_tokens / PREFILL_TOK_PER_S
            + event.out_tokens / DECODE_TOK_PER_S
        )
        per_unit[event.unit_id] = per_unit.get(event.unit_id, 0.0) + work
    return max(30.0, 1.5 * max(per_unit.values(), default=30.0))


def _group(events: list[TraceEvent]) -> dict[str, list[TraceEvent]]:
    grouped: dict[str, list[TraceEvent]] = {}
    for event in events:
        grouped.setdefault(event.unit_id, []).append(event)
    return grouped


def _agent_for(events: list[TraceEvent]) -> AgentFn:
    async def run(handle: AgentHandle) -> None:
        for event in events:
            if event.kind == "reason":
                await handle.reason(
                    prompt_tokens=max(1, event.prompt_tokens), out_tokens=event.out_tokens
                )
            else:
                spec = ToolSpec(
                    event.name, "python", expected_duration_s=max(0.1, event.tool_s)
                )
                await handle.call_tool(spec, {"code": "pass"})

    return run


def _caps(capacity_gib: float) -> BackendCaps:
    """Same model constants as ``demo.caps`` so replay and experiments are comparable."""
    return BackendCaps(
        capacity_bytes=int(capacity_gib * 1024**3),
        block_tokens=BLOCK_TOKENS,
        bytes_per_token=BYTES_PER_TOKEN,
        prefill_tok_per_s=PREFILL_TOK_PER_S,
        decode_tok_per_s=DECODE_TOK_PER_S,
        mem_bw_bps=int(12 * 1024**3),
        supports_offload=True,
    )


async def _run_once(
    events: list[TraceEvent],
    policy: str,
    *,
    capacity_gib: float,
    seed: int,
    slo_s: float,
    arrival_rate_per_s: float | None,
    prefix_hit_ratio: float,
) -> RunReport:
    clock = VirtualClock()
    clock.start_driver()
    caps = _caps(capacity_gib)
    backend = MockBackend(caps=caps, clock=clock, prefix_hit_ratio=prefix_hit_ratio)
    runtime = Runtime(
        backend=backend,
        clock=clock,
        budget=Budget(total_bytes=caps.capacity_bytes, headroom_frac=0.05),
        policy=build_policy(policy),
        config=RuntimeConfig(run_id=f"replay-{policy}-{seed}", simulate_tools=True),
    )
    await runtime.start()
    try:
        rng = random.Random(seed)
        tasks: list[asyncio.Task[TaskOutcome]] = []
        for index, (unit_id, unit_events) in enumerate(_group(events).items()):
            if arrival_rate_per_s is not None and index > 0:
                await clock.wait(rng.expovariate(arrival_rate_per_s))
            tasks.append(
                asyncio.create_task(runtime.run_task(unit_id, _agent_for(unit_events), slo_s=slo_s))
            )
        await asyncio.gather(*tasks)
    finally:
        await runtime.shutdown()
        clock.stop_driver()
    return MetricsReducer.replay(runtime.journal)


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def replay_compare(
    events: list[TraceEvent],
    *,
    policies: tuple[str, ...] = ("B0", "B2"),
    capacity_gib: float = 2.0,
    seeds: tuple[int, ...] = (0, 1, 2),
    arrival_rate_per_s: float | None = None,
    prefix_hit_ratio: float = 0.0,
    slo_s: float | None = None,
) -> ReplayResult:
    """Compare policies on the caller's own workload. All numbers are [L1]."""
    deadline = estimate_slo(events) if slo_s is None else slo_s
    by_policy: dict[str, dict[str, float]] = {}
    for policy in policies:
        rows: list[dict[str, float]] = []
        for seed in seeds:
            report = asyncio.run(
                _run_once(
                    events,
                    policy,
                    capacity_gib=capacity_gib,
                    seed=seed,
                    slo_s=deadline,
                    arrival_rate_per_s=arrival_rate_per_s,
                    prefix_hit_ratio=prefix_hit_ratio,
                )
            )
            rows.append(report.summary_row())
        by_policy[policy] = _mean_rows(rows)
    return ReplayResult(
        evidence="[L1] trace replay / deterministic simulation (not a real-GPU measurement)",
        trace_events=len(events),
        units=len(_group(events)),
        capacity_gib=capacity_gib,
        seeds=seeds,
        slo_s=deadline,
        prefix_hit_ratio=prefix_hit_ratio,
        prefix=prefix_overlap(events),
        policies=by_policy,
    )


def format_report(result: ReplayResult) -> str:
    """Table with the mandatory [L1] header and seed / C / workload footnotes."""
    prefix_hit = (
        "off (0.0)" if result.prefix_hit_ratio == 0.0 else f"{result.prefix_hit_ratio:.2f} [L2]"
    )
    lines = [
        result.evidence,
        f"trace_events={result.trace_events} units={result.units} "
        f"C={result.capacity_gib}GiB seeds={','.join(str(s) for s in result.seeds)} "
        f"slo={result.slo_s:.1f}s prefix_hit={prefix_hit}",
        f"prefix overlap: logical={result.prefix.logical_blocks} blocks, "
        f"distinct={result.prefix.distinct_blocks} blocks, "
        f"dedup potential={result.prefix.label}",
        "",
        f"{'policy':<8}{'done':>7}{'slo':>8}{'goodput':>10}{'p99_ms':>10}"
        f"{'thrash':>9}{'recomp':>10}{'kv_waste':>11}",
    ]
    for policy, row in result.policies.items():
        lines.append(
            f"{policy:<8}{int(row.get('completed', 0)):>7}{row.get('slo_rate', 0):>8.3f}"
            f"{row.get('goodput', 0):>10.4f}{row.get('p99_ms', 0):>10.0f}"
            f"{row.get('thrash', 0):>9.2f}{int(row.get('recompute_tok', 0)):>10}"
            f"{row.get('kv_waste_mib_s', 0):>11.0f}"
        )
    return "\n".join(lines)
