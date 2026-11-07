"""Experiment CLI. Every table carries an [L1]/[L2] header (spec 12.5)."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.leases import Budget
from phaseweaver.metrics import MetricsReducer, RunReport
from phaseweaver.policy import Scheduler
from phaseweaver.runtime import Runtime, RuntimeConfig
from phaseweaver.tools.base import ToolSpec
from phaseweaver.types import ForkMode

BYTES_PER_TOKEN = 57344  # Qwen2.5-7B bf16 KV per token
BLOCK_TOKENS = 16


@dataclass
class Workload:
    name: str
    agents: int
    rounds: int
    context_tokens: int
    out_tokens: int
    slo_s: float
    seed: int
    tool_p50_s: float
    tool_sigma: float
    rate_per_s: float
    fanout: int = 0
    sub_rounds: int = 0
    fork_mode: str = "share"
    sub_context_tokens: int = 0
    greedy_factor: float = 1.0
    #: none | handoff | relay  (inherit_prefix kept as a legacy boolean alias)
    inherit_mode: str = "none"
    #: 2 means each subagent also delegates to a grandchild
    nested_depth: int = 1
    #: R6: 0 keeps tools uniform; >0 splits agents into short/long tool groups
    #: (even index: * (1+h), odd index: * (1-h)) so phase alignment has something
    #: to align.
    tool_hetero: float = 0.0

    @classmethod
    def load(cls, path: Path) -> Workload:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        w, t, a = data["workload"], data["tool"], data["arrival"]
        return cls(
            name=str(w["name"]),
            agents=int(w["agents"]),
            rounds=int(w["rounds"]),
            context_tokens=int(w["context_tokens"]),
            out_tokens=int(w["out_tokens"]),
            slo_s=float(w["slo_s"]),
            seed=int(w["seed"]),
            tool_p50_s=float(t["p50_s"]),
            tool_sigma=float(t["sigma"]),
            rate_per_s=float(a["rate_per_s"]),
            fanout=int(w.get("fanout", 0)),
            sub_rounds=int(w.get("sub_rounds", 0)),
            fork_mode=str(w.get("fork_mode", "share")),
            sub_context_tokens=int(w.get("sub_context_tokens", 0)),
            greedy_factor=float(w.get("greedy_factor", 1.0)),
            inherit_mode=str(
                w.get("inherit_mode")
                or ("handoff" if w.get("inherit_prefix") in (True, "true") else "none")
            ),
            nested_depth=int(w.get("nested_depth", 1)),
            tool_hetero=float(t.get("hetero", 0.0)),
        )


def caps(capacity_gib: float = 8.0) -> BackendCaps:
    return BackendCaps(
        capacity_bytes=int(capacity_gib * 1024**3),
        block_tokens=BLOCK_TOKENS,
        bytes_per_token=BYTES_PER_TOKEN,
        prefill_tok_per_s=3200,
        decode_tok_per_s=45,
        mem_bw_bps=int(12 * 1024**3),
        supports_offload=True,
    )


def build_policy(name: str) -> Scheduler:
    """Single source of truth: policy names are owned by the facade, not the CLI.

    (Keeping a second copy here is how `align2` silently became "unknown policy".)
    """
    from phaseweaver.app import build_policy as facade_build_policy

    return facade_build_policy(name)


async def run_one(
    workload: Workload,
    policy_name: str,
    *,
    capacity_gib: float = 8.0,
    journal_path: Path | None = None,
) -> RunReport:
    clock = VirtualClock()
    clock.start_driver()
    backend_caps = caps(capacity_gib)
    backend = MockBackend(caps=backend_caps, clock=clock)
    runtime = Runtime(
        backend=backend,
        clock=clock,
        budget=Budget(total_bytes=backend_caps.capacity_bytes, headroom_frac=0.05),
        policy=build_policy(policy_name),
        config=RuntimeConfig(run_id=f"{workload.name}-{policy_name}", journal_path=journal_path),
    )
    await runtime.start()
    rng = random.Random(workload.seed)
    tools = [
        ToolSpec(
            f"tool-{i}",
            "python",
            expected_duration_s=max(
                0.1, rng.lognormvariate(0.0, workload.tool_sigma) * workload.tool_p50_s
            ),
        )
        for i in range(workload.rounds)
    ]

    async def child(handle: Any, greedy: bool = False) -> None:
        base = workload.sub_context_tokens or workload.context_tokens // 2
        base = int(base * (workload.greedy_factor if greedy else 1.0))
        # Same conversation length in both arms; only the sharing differs, otherwise
        # the comparison would just measure a smaller prompt.
        prompt = workload.context_tokens + base
        for _ in range(workload.sub_rounds):
            await handle.reason(prompt_tokens=prompt, out_tokens=workload.out_tokens)
            await handle.call_tool(
                ToolSpec("sub-tool", "python", expected_duration_s=workload.tool_p50_s),
                {"code": "pass"},
            )
            prompt += workload.out_tokens
        if workload.nested_depth >= 2:
            await handle.fork(
                "grandchild", _grandchild, inherit=workload.inherit_mode
            )

    async def _grandchild(handle: Any) -> None:
        await handle.reason(
            prompt_tokens=workload.context_tokens // 2, out_tokens=workload.out_tokens
        )

    async def agent(handle: Any, specs: list[ToolSpec] | None = None) -> None:
        prompt = workload.context_tokens
        for spec in specs or tools:
            await handle.reason(prompt_tokens=prompt, out_tokens=workload.out_tokens)
            await handle.call_tool(spec, {"code": "pass"})
            prompt += workload.out_tokens  # a conversation grows every round
        if workload.fanout > 0:
            # Fork a group of subagents: the parent parks and waits (WAITING),
            # children share the parent's budget (SHARE) or get a hard cap (ISOLATE).
            mode = ForkMode.ISOLATE if workload.fork_mode == "isolate" else ForkMode.SHARE
            if workload.inherit_mode == "relay":
                # Relay: children run one after another, each inheriting the KV from
                # the previous one through the parent - every delegate starts warm.
                for i in range(workload.fanout):
                    await handle.fork(
                        f"sub-{i}",
                        lambda h, g=(i == 0): child(h, g),
                        mode=mode,
                        inherit=workload.inherit_mode,
                    )
            else:
                await asyncio.gather(
                    *(
                        handle.fork(
                            f"sub-{i}",
                            lambda h, g=(i == 0): child(h, g),
                            mode=mode,
                            inherit=workload.inherit_mode,
                        )
                        for i in range(workload.fanout)
                    )
                )
            await handle.reason(prompt_tokens=prompt, out_tokens=workload.out_tokens)

    pending: list[asyncio.Task[Any]] = []
    for i in range(workload.agents):
        if workload.rate_per_s > 0:
            await clock.wait(rng.expovariate(workload.rate_per_s))
        if workload.tool_hetero > 0:
            scale = 1.0 + workload.tool_hetero if i % 2 == 0 else max(
                0.1, 1.0 - workload.tool_hetero
            )
            specs = [
                ToolSpec(
                    spec.name,
                    spec.executor,
                    expected_duration_s=(spec.expected_duration_s or workload.tool_p50_s)
                    * scale,
                )
                for spec in tools
            ]

            async def tuned(handle: Any, chosen: list[ToolSpec] = specs) -> None:
                await agent(handle, chosen)

            job = runtime.run_task(f"task-{i}", tuned, slo_s=workload.slo_s)
        else:
            job = runtime.run_task(f"task-{i}", agent, slo_s=workload.slo_s)
        pending.append(asyncio.create_task(job))
    await asyncio.gather(*pending)
    await runtime.shutdown()
    clock.stop_driver()
    return MetricsReducer.replay(runtime.journal)


HEADER = (
    f"{'policy':<7}{'done':>6}{'slo':>8}{'goodput':>9}{'p50':>9}"
    f"{'p95':>9}{'p99':>9}{'thrash':>8}{'recomp':>9}{'kv_waste':>10}"
)


def print_row(policy: str, row: dict[str, float]) -> None:
    print(
        f"{policy:<7}{int(row['completed']):>6}{row['slo_rate']:>8.3f}{row['goodput']:>9.3f}"
        f"{row['p50_ms']:>9.0f}{row['p95_ms']:>9.0f}{row['p99_ms']:>9.0f}"
        f"{row['thrash']:>8.2f}{int(row['recompute_tok']):>9}{row['kv_waste_mib_s']:>10.0f}"
    )


def cmd_compare(args: argparse.Namespace) -> int:
    workload = Workload.load(Path(args.workload))
    print(
        f"[L1] {workload.name} capacity={args.capacity_gib}GiB slo={workload.slo_s}s "
        f"seed={workload.seed}"
    )
    print(HEADER)
    rows = []
    for policy in args.policies.split(","):
        report = asyncio.run(run_one(workload, policy, capacity_gib=args.capacity_gib))
        row = report.summary_row()
        rows.append({"policy": policy, **row})
        print_row(policy, row)
    print(json.dumps(rows, ensure_ascii=False))
    return 0


def cmd_burst(args: argparse.Namespace) -> int:
    workload = Workload.load(Path(args.workload))
    base_rate = workload.rate_per_s
    print(f"[L1] burst x{args.multiplier} on {workload.name} capacity={args.capacity_gib}GiB")
    print(f"{'policy':<7}{'done':>6}{'slo':>8}{'p99':>9}{'thrash':>8}{'recomp':>9}")
    for policy in args.policies.split(","):
        workload.rate_per_s = base_rate * args.multiplier
        report = asyncio.run(run_one(workload, policy, capacity_gib=args.capacity_gib))
        print(
            f"{policy:<7}{report.completed:>6}{report.slo_attainment_rate:>8.3f}"
            f"{report.latency_p99_ms:>9.0f}{report.thrash_rate:>8.2f}"
            f"{report.recompute_tokens:>9}"
        )
        workload.rate_per_s = base_rate
    return 0


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(var) / m if m else 0.0


def cmd_sweep(args: argparse.Namespace) -> int:
    workload = Workload.load(Path(args.workload))
    base_seed = workload.seed
    print(f"[L2] sweep {workload.name} seeds={args.seeds}")
    print(f"{'cap':>5}{'policy':<7}{'goodput':>9}{'p99':>9}{'slo':>8}{'thrash':>8}")
    for gib in [float(x) for x in args.capacities.split(",")]:
        for policy in args.policies.split(","):
            goodputs: list[float] = []
            p99s: list[float] = []
            slos: list[float] = []
            thrash: list[float] = []
            for seed in range(args.seeds):
                workload.seed = seed
                report = asyncio.run(
                    run_one(workload, policy, capacity_gib=gib)
                )
                goodputs.append(report.goodput)
                p99s.append(report.latency_p99_ms)
                slos.append(report.slo_attainment_rate)
                thrash.append(report.thrash_rate)
            print(
                f"{gib:>5.0f}{policy:<7}{sum(goodputs) / len(goodputs):>9.4f}"
                f"{sum(p99s) / len(p99s):>9.0f}{sum(slos) / len(slos):>8.3f}"
                f"{sum(thrash) / len(thrash):>8.2f}"
            )
            print(f"        goodput_cv={_cv(goodputs):.3f} p99_cv={_cv(p99s):.3f}")
    workload.seed = base_seed
    return 0


def cmd_optimal(args: argparse.Namespace) -> int:
    from phaseweaver.oracle import AlphaModel, lower_bound_makespan

    workload = Workload.load(Path(args.workload))
    report = asyncio.run(run_one(workload, args.policy, capacity_gib=args.capacity_gib))
    c = caps(args.capacity_gib)
    model = AlphaModel(
        prefill_tok_per_s=c.prefill_tok_per_s,
        decode_tok_per_s=c.decode_tok_per_s,
        alpha=c.contention_alpha,
        capacity_bytes=int(c.capacity_bytes * 0.95),  # same headroom as the run
        bytes_per_token=c.bytes_per_token,
    )
    prefill = workload.context_tokens + workload.out_tokens
    bound_s = lower_bound_makespan(
        agents=workload.agents,
        rounds=workload.rounds,
        prefill_tokens=prefill,
        decode_tokens=workload.out_tokens,
        tool_s=workload.tool_p50_s,
        model=model,
    )
    observed = report.wall_clock_s
    gap = (observed - bound_s) / bound_s * 100 if bound_s else 0.0
    k = model.max_streams(prefill, workload.agents)
    print(
        f"[L2] optimality policy={args.policy} observed={observed:.1f}s "
        f"bound={bound_s:.1f}s gap={gap:.1f}% max_streams={k} "
        f"mean_conc={report.mean_concurrency:.2f} bound_kind=duty_aware_lower_bound"
    )
    return 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def cmd_scenarios(args: argparse.Namespace) -> int:
    """Run each concrete scenario with its recommended configuration vs the baseline."""
    from phaseweaver.scenarios import SCENARIOS

    print(f"[L1] scenario sweep capacity={args.capacity_gib}GiB seeds={args.seeds}")
    print(
        f"{'scenario':<12}{'config':<18}{'goodput':>9}{'p99':>9}{'slo':>8}"
        f"{'kv_waste':>10}{'failed':>8}"
    )
    for scenario in SCENARIOS:
        workload = Workload.load(scenario.path())
        if scenario.recommended == "auto":
            setups = [
                ("baseline(B0)", "B0", "none"),
                ("B2(fixed)", "B2", "none"),
                ("autopilot", "auto", "none"),
            ]
        else:
            setups = [
                ("baseline(B0)", "B0", "none"),
                ("B2(fixed)", "B2", "none"),
                ("handoff", "B2", "handoff"),
            ]
        for label, policy, inherit in setups:
            goodput: list[float] = []
            p99: list[float] = []
            slo: list[float] = []
            waste: list[float] = []
            failed: list[float] = []
            for seed in range(args.seeds):
                workload.seed = seed
                workload.inherit_mode = inherit
                report = asyncio.run(
                    run_one(workload, policy, capacity_gib=args.capacity_gib)
                )
                goodput.append(report.goodput)
                p99.append(report.latency_p99_ms / 1000)
                slo.append(report.slo_attainment_rate)
                waste.append(report.kv_occupancy_waste_bytes_s / 1024 / 1024)
                failed.append(float(report.failed))
            _mean_ = _mean
            print(
                f"{scenario.key:<12}{label:<18}{_mean_(goodput):>9.4f}{_mean_(p99):>9.1f}"
                f"{_mean_(slo):>8.3f}{_mean_(waste):>10.0f}{_mean_(failed):>8.1f}"
            )
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    from phaseweaver.journal import EventJournal
    from phaseweaver.timeline import build_timeline

    journal = EventJournal.load(Path(args.journal))
    timeline = build_timeline(journal)
    if args.out:
        timeline.to_jsonl(Path(args.out))
        print(f"wrote {len(timeline.spans)} spans to {args.out}")
    else:
        for span in timeline.spans:
            print(
                f"{span.unit_id:<40} {span.kind:<7} {span.name:<10} "
                f"{span.start:7.2f} -> {span.end:7.2f} ({span.duration:6.2f}s)"
            )
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from phaseweaver.journal import EventJournal
    from phaseweaver.timeline import build_timeline
    from phaseweaver.trace import format_trace, trace_unit

    journal = EventJournal.load(Path(args.journal))
    timeline = build_timeline(journal)
    units = (
        [
            event.unit_id
            for event in journal.by_type("unit_created")
            if event.payload.get("kind") in ("agent", "subagent") and event.unit_id
        ]
        if args.all
        else [args.unit]
    )
    for unit_id in units:
        if unit_id is None:
            continue
        trace = trace_unit(journal, unit_id, timeline)
        if args.all:
            print(
                f"{unit_id:<44} latency={trace.latency_s:7.2f}s "
                f"dominant={trace.dominant[0]:<9} parks={trace.park_count:<3} "
                f"recompute={trace.recompute_tokens}"
            )
        else:
            print(format_trace(trace))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phaseweaver")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cmp_p = sub.add_parser("compare", help="E1: baseline comparison")
    cmp_p.add_argument("--workload", required=True)
    cmp_p.add_argument("--policies", default="B0,B1,B2a,B2")
    cmp_p.add_argument("--capacity-gib", type=float, default=8.0)

    bu = sub.add_parser("burst", help="E2: burst load")
    bu.add_argument("--workload", required=True)
    bu.add_argument("--policies", default="B0,B1,B2a,B2")
    bu.add_argument("--capacity-gib", type=float, default=8.0)
    bu.add_argument("--multiplier", type=float, default=3.0)

    sw = sub.add_parser("sweep", help="E3/E4: capacity sweep across seeds")
    sw.add_argument("--workload", required=True)
    sw.add_argument("--capacities", default="2,4,8,16")
    sw.add_argument("--policies", default="B0,B1,B2")
    sw.add_argument("--seeds", type=int, default=20)

    op = sub.add_parser("optimal", help="E5: optimality gap vs B4")
    op.add_argument("--workload", required=True)
    op.add_argument("--policy", default="B2")
    op.add_argument("--capacity-gib", type=float, default=8.0)

    sc = sub.add_parser("scenarios", help="run the concrete usage scenarios")
    sc.add_argument("--capacity-gib", type=float, default=2.0)
    sc.add_argument("--seeds", type=int, default=3)

    tl = sub.add_parser("timeline", help="phase timeline from a saved journal")
    tl.add_argument("--journal", required=True)
    tl.add_argument("--out", default=None, help="write JSONL here instead of printing")

    tr = sub.add_parser("trace", help="post-mortem for one unit (or all)")
    tr.add_argument("--journal", required=True)
    tr.add_argument("--unit", default=None)
    tr.add_argument("--all", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "burst":
        return cmd_burst(args)
    if args.cmd == "sweep":
        return cmd_sweep(args)
    if args.cmd == "optimal":
        return cmd_optimal(args)
    if args.cmd == "scenarios":
        return cmd_scenarios(args)
    if args.cmd == "timeline":
        return cmd_timeline(args)
    if args.cmd == "trace":
        return cmd_trace(args)
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
