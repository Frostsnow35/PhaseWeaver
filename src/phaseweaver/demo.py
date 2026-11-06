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
from phaseweaver.policy import PinAlwaysPolicy, Scheduler, TtlEvictPolicy, YieldOnToolPolicy
from phaseweaver.runtime import Runtime, RuntimeConfig
from phaseweaver.scheduler import SchedulerConfig, SystemScheduler
from phaseweaver.tools.base import ToolSpec

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
    if name == "B0":
        return PinAlwaysPolicy()
    if name == "B1":
        return TtlEvictPolicy(ttl_s=5.0)
    if name == "B2a":
        return YieldOnToolPolicy()
    if name == "B2":
        return SystemScheduler(SchedulerConfig())
    raise ValueError(f"unknown policy {name}")


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
        budget=Budget(total_bytes=backend_caps.capacity_bytes, headroom_frac=0.0),
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

    async def agent(handle: Any) -> None:
        for spec in tools:
            await handle.reason(
                prompt_tokens=workload.context_tokens, out_tokens=workload.out_tokens
            )
            await handle.call_tool(spec, {"code": "pass"})

    pending: list[asyncio.Task[Any]] = []
    for i in range(workload.agents):
        if workload.rate_per_s > 0:
            await clock.wait(rng.expovariate(workload.rate_per_s))
        pending.append(
            asyncio.create_task(runtime.run_task(f"task-{i}", agent, slo_s=workload.slo_s))
        )
    await asyncio.gather(*pending)
    await runtime.shutdown()
    clock.stop_driver()
    return MetricsReducer.replay(runtime.journal)


HEADER = (
    f"{'policy':<7}{'done':>6}{'slo':>8}{'goodput':>9}{'p50':>9}"
    f"{'p95':>9}{'p99':>9}{'thrash':>8}{'recomp':>9}"
)


def print_row(policy: str, row: dict[str, float]) -> None:
    print(
        f"{policy:<7}{int(row['completed']):>6}{row['slo_rate']:>8.3f}{row['goodput']:>9.3f}"
        f"{row['p50_ms']:>9.0f}{row['p95_ms']:>9.0f}{row['p99_ms']:>9.0f}"
        f"{row['thrash']:>8.2f}{int(row['recompute_tok']):>9}"
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
    from phaseweaver.oracle import OracleBudget, brute_force_min_makespan

    workload = Workload.load(Path(args.workload))
    report = asyncio.run(run_one(workload, args.policy, capacity_gib=args.capacity_gib))
    reasoning_s = workload.context_tokens / 3200 + workload.out_tokens / 45
    bound_s = brute_force_min_makespan(
        tasks=[workload.rounds] * workload.agents,
        reasoning_s=[reasoning_s] * workload.agents,
        tool_s=[workload.tool_p50_s] * workload.agents,
        budget=OracleBudget(
            max_resident=max(1, int(args.capacity_gib * 1024**3 / (2048 * BYTES_PER_TOKEN)))
        ),
    )
    observed = report.wall_clock_s
    gap = (observed - bound_s) / bound_s * 100 if bound_s else 0.0
    print(
        f"[L2] optimality policy={args.policy} observed={observed:.1f}s "
        f"bound={bound_s:.1f}s gap={gap:.1f}% bound_kind=heuristic_lower_bound"
    )
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

    args = parser.parse_args(argv)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "burst":
        return cmd_burst(args)
    if args.cmd == "sweep":
        return cmd_sweep(args)
    if args.cmd == "optimal":
        return cmd_optimal(args)
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
