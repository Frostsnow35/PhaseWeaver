"""[L2] Capacity-amplification upper bound for the prefix-sharing question.

The ``prefix_hit_ratio`` probe (scripts/probe_prefix.py) only discounts
*prefill time*. The other half of breakthrough A is that deduplicating shared
prefix blocks would let the same physical memory hold more logical contexts.
That is exactly equivalent to a larger effective capacity, so this script
sweeps ``C = C_base * (1 + k)`` with the unchanged runtime - no kernel change,
no invented block accounting.

k -> f/(1-f) where f is the real shared-prefix fraction. Synthetic traces give
f ~ 0.94 (k ~ 16, unrealistic); real agents usually share 10-40% of tokens
(k = 0.11-0.67). So the k=0.5 column is the realistic ceiling.

Every row is [L2] parameter sensitivity on the deterministic simulator, not a
real-GPU measurement. Raw rows go to runs/capacity-probe.json (gitignored).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from phaseweaver.demo import Workload, run_one

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = (
    "w2.toml",
    "w3-queue.toml",
    "w5-fanout.toml",
    "w8-handoff.toml",
    "w9-fanout.toml",
    "w10-nested.toml",
)
AMPLIFICATIONS = (0.0, 0.25, 0.5, 1.0, 2.0)
SEEDS = (0, 1, 2)
BASE_CAPACITY_GIB = 2.0
POLICY = "B2"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    rows: list[dict[str, object]] = []
    print(
        f"[L2] capacity amplification upper bound  C_base={BASE_CAPACITY_GIB}GiB "
        f"policy={POLICY} seeds={','.join(map(str, SEEDS))}"
    )
    header = f"{'workload':<18}" + "".join(f"{'k=' + str(k):>10}" for k in AMPLIFICATIONS)
    print(header + f"{'gain@k=0.5':>12}{'gain@k=1':>10}")
    for name in WORKLOADS:
        path = ROOT / "examples" / "workloads" / name
        curve: list[float] = []
        for k in AMPLIFICATIONS:
            goodputs: list[float] = []
            for seed in SEEDS:
                workload = Workload.load(path)
                workload.seed = seed
                report = asyncio.run(
                    run_one(
                        workload,
                        POLICY,
                        capacity_gib=BASE_CAPACITY_GIB * (1 + k),
                    )
                )
                goodputs.append(report.summary_row()["goodput"])
            value = _mean(goodputs)
            curve.append(value)
            rows.append(
                {
                    "workload": name,
                    "amplification": k,
                    "capacity_gib": BASE_CAPACITY_GIB * (1 + k),
                    "goodput": value,
                }
            )
        gain_half = (curve[2] - curve[0]) / curve[0] * 100 if curve[0] else 0.0
        gain_full = (curve[3] - curve[0]) / curve[0] * 100 if curve[0] else 0.0
        print(
            f"{name:<18}" + "".join(f"{v:>10.4f}" for v in curve)
            + f"{gain_half:>11.1f}%{gain_full:>9.1f}%"
        )

    out = ROOT / "runs" / "capacity-probe.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "evidence": "[L2] parameter sensitivity / deterministic simulation",
                "caveat": "effective-capacity sweep is an upper bound; no block dedup implemented",
                "base_capacity_gib": BASE_CAPACITY_GIB,
                "policy": POLICY,
                "seeds": list(SEEDS),
                "amplifications": list(AMPLIFICATIONS),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[L2] raw rows -> {out}")
    print("[L2] read k=0.5 as the realistic ceiling (shared-prefix fraction ~33%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
