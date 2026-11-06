"""Re-run the key experiment matrix and print markdown tables.

Usage: python scripts/run_experiments.py [--seeds N] [--quick]

All output is mock-simulation data: label it [L1] / [L2] in any report.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phaseweaver.demo import Workload, run_one  # noqa: E402

WORKLOADS = Path("examples/workloads")


def measure(
    workload: Workload, policy: str, capacity_gib: float, seeds: int
) -> dict[str, float]:
    goodput: list[float] = []
    p99: list[float] = []
    slo: list[float] = []
    jain: list[float] = []
    kv_waste: list[float] = []
    failed: list[float] = []
    for seed in range(seeds):
        workload.seed = seed
        report = asyncio.run(run_one(workload, policy, capacity_gib=capacity_gib))
        goodput.append(report.goodput)
        p99.append(report.latency_p99_ms / 1000)
        slo.append(report.slo_attainment_rate)
        jain.append(report.jain_fairness)
        kv_waste.append(report.kv_occupancy_waste_bytes_s / 1024 / 1024)
        failed.append(float(report.failed))

    def mean(values: list[float]) -> float:
        return statistics.mean(values)

    return {
        "goodput": mean(goodput),
        "p99_s": mean(p99),
        "slo": mean(slo),
        "jain": mean(jain),
        "kv_waste": mean(kv_waste),
        "failed": mean(failed),
    }


def row(label: str, m: dict[str, float]) -> str:
    return (
        f"| {label} | {m['goodput']:.4f} | {m['p99_s']:.1f} | {m['slo']:.3f} "
        f"| {m['jain']:.3f} | {m['kv_waste']:.0f} | {m['failed']:.1f} |"
    )


HEADER = (
    "| 策略 | goodput | p99 (s) | SLO | Jain | kv_waste | failed |\n"
    "|---|---|---|---|---|---|---|"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="only 2/4 GiB and fewer runs")
    args = parser.parse_args()
    seeds = args.seeds

    print(f"# 复现实验 seeds={seeds}\n")

    print("## E3 容量扫描(w3 队列场景)[L2]\n")
    for cap in ([2.0, 4.0] if args.quick else [2.0, 4.0, 8.0]):
        print(f"\n### {cap:.0f} GiB\n")
        print(HEADER)
        workload = Workload.load(WORKLOADS / "w3-queue.toml")
        for policy in ("B0", "B1", "B2"):
            print(row(policy, measure(workload, policy, cap, seeds)))

    print("\n## K7 子 Agent 扇出(w5,2 GiB)[L1]\n")
    print(HEADER)
    fanout = Workload.load(WORKLOADS / "w5-fanout.toml")
    for policy in ("B0", "B2"):
        print(row(policy, measure(fanout, policy, 2.0, seeds)))

    print("\n## K9/K11 贪婪子 Agent(w6-12,2 GiB)[L1]\n")
    print(HEADER)
    for name in ("w6-greedy-share-12.toml", "w6-greedy-isolate-12.toml"):
        w = Workload.load(WORKLOADS / name)
        label = "SHARE" if "share" in name else "ISOLATE(compacted)"
        print(row(label, measure(w, "B2", 2.0, seeds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
