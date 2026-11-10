"""Command-line surface for the experiment harness.

``demo.py`` owns the workload model and the reproducible runner; this module
owns argument parsing, table formatting and subcommand dispatch. Split out so
both files stay under the 400-line limit (AGENTS.md section 5). Every table
carries an [L1]/[L2] header (spec 12.5).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from phaseweaver.demo import Workload, caps, run_one

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
    if args.prefix_hit_ratio:
        workload.prefix_hit_ratio = args.prefix_hit_ratio
    level = "[L2]" if workload.prefix_hit_ratio > 0 else "[L1]"
    print(
        f"{level} {workload.name} capacity={args.capacity_gib}GiB slo={workload.slo_s}s "
        f"seed={workload.seed} prefix_hit={workload.prefix_hit_ratio}"
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
                report = asyncio.run(run_one(workload, policy, capacity_gib=gib))
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
            print(
                f"{scenario.key:<12}{label:<18}{_mean(goodput):>9.4f}{_mean(p99):>9.1f}"
                f"{_mean(slo):>8.3f}{_mean(waste):>10.0f}{_mean(failed):>8.1f}"
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


def cmd_replay(args: argparse.Namespace) -> int:
    """Compare policies on the caller's own agent trace; no GPU and no engine needed."""
    from phaseweaver.replay import format_report, load_trace, replay_compare

    events = load_trace(Path(args.trace))
    seeds = tuple(int(seed) for seed in args.seeds.split(","))
    result = replay_compare(
        events,
        policies=tuple(args.policies.split(",")),
        capacity_gib=args.capacity_gib,
        seeds=seeds,
        arrival_rate_per_s=args.arrival_rate,
        prefix_hit_ratio=args.prefix_hit_ratio,
        slo_s=args.slo_s,
    )
    print(format_report(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phaseweaver")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cmp_p = sub.add_parser("compare", help="E1: baseline comparison")
    cmp_p.add_argument("--workload", required=True)
    cmp_p.add_argument("--policies", default="B0,B1,B2a,B2")
    cmp_p.add_argument("--capacity-gib", type=float, default=8.0)
    cmp_p.add_argument(
        "--prefix-hit-ratio",
        type=float,
        default=0.0,
        help="[L2] experimental prefix-cache upper bound, 0.0 = original model",
    )

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

    rp = sub.add_parser("replay", help="compare policies on your own agent trace (JSONL)")
    rp.add_argument("--trace", required=True)
    rp.add_argument("--policies", default="B0,B2")
    rp.add_argument("--capacity-gib", type=float, default=2.0)
    rp.add_argument("--seeds", default="0,1,2")
    rp.add_argument("--arrival-rate", type=float, default=None)
    rp.add_argument("--prefix-hit-ratio", type=float, default=0.0)
    rp.add_argument("--slo-s", type=float, default=None)

    args = parser.parse_args(argv)
    handlers = {
        "compare": cmd_compare,
        "burst": cmd_burst,
        "sweep": cmd_sweep,
        "optimal": cmd_optimal,
        "scenarios": cmd_scenarios,
        "timeline": cmd_timeline,
        "trace": cmd_trace,
        "replay": cmd_replay,
    }
    handler = handlers.get(args.cmd)
    if handler is None:
        raise SystemExit(f"unknown command {args.cmd}")
    return handler(args)
