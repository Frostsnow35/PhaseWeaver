"""[L2] Prefix-hit sensitivity probe: is breakthrough A worth implementing?

``prefix_hit_ratio`` is an explicit *assumption* that a shared prefix cache
would cover a fraction of the prefill work charged to a recompute. It is an
upper bound, not a mechanism: neither refcounting nor block dedup exists yet.
This script only answers "if A worked this well, how much would B2 gain?".

Gate (docs/plans/2026-11-08-strategy-v2.md P0 follow-up):
  PASS  if the goodput curve is monotone in h and gain at h=0.5 is >= 5%
  STOP  otherwise -> do not touch the resource ledger

Every table carries [L2], seeds, C and workload parameters. Raw rows are
written to runs/prefix-probe.json (gitignored), never into the repository.
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
H_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2)
POLICIES = ("B0", "B2")
CAPACITY_GIB = 2.0
GAIN_THRESHOLD = 0.05


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    rows: list[dict[str, object]] = []
    print(f"[L2] prefix-hit sensitivity  C={CAPACITY_GIB}GiB  seeds={','.join(map(str, SEEDS))}")
    print(
        f"{'workload':<18}{'h':>6}{'policy':<7}{'goodput':>10}{'p99_s':>9}"
        f"{'slo':>7}{'recomp':>10}{'kv_waste':>11}"
    )
    for name in WORKLOADS:
        path = ROOT / "examples" / "workloads" / name
        for hit in H_VALUES:
            for policy in POLICIES:
                goodputs: list[float] = []
                p99s: list[float] = []
                slos: list[float] = []
                recomputes: list[float] = []
                wastes: list[float] = []
                for seed in SEEDS:
                    workload = Workload.load(path)
                    workload.seed = seed
                    workload.prefix_hit_ratio = hit
                    report = asyncio.run(run_one(workload, policy, capacity_gib=CAPACITY_GIB))
                    row = report.summary_row()
                    goodputs.append(row["goodput"])
                    p99s.append(row["p99_ms"] / 1000)
                    slos.append(row["slo_rate"])
                    recomputes.append(row["recompute_tok"])
                    wastes.append(row["kv_waste_mib_s"])
                entry = {
                    "workload": name,
                    "prefix_hit_ratio": hit,
                    "policy": policy,
                    "goodput": _mean(goodputs),
                    "p99_s": _mean(p99s),
                    "slo_rate": _mean(slos),
                    "recompute_tok": _mean(recomputes),
                    "kv_waste_mib_s": _mean(wastes),
                }
                rows.append(entry)
                print(
                    f"{name:<18}{hit:>6.2f}{policy:<7}{entry['goodput']:>10.4f}"
                    f"{entry['p99_s']:>9.1f}{entry['slo_rate']:>7.3f}"
                    f"{entry['recompute_tok']:>10.0f}{entry['kv_waste_mib_s']:>11.0f}"
                )

    print("\n[L2] gate: B2 goodput gain from h=0 -> h=0.5, and curve monotonicity")
    print(f"{'workload':<18}{'gp@h0':>9}{'gp@h0.5':>10}{'gain':>9}{'monotone':>10}{'verdict':>9}")
    verdicts: dict[str, str] = {}
    for name in WORKLOADS:
        curve = [
            float(r["goodput"])
            for r in rows
            if r["workload"] == name and r["policy"] == "B2"
        ]
        tolerance = 0.02 * max(curve)
        monotone = all(
            b >= a - tolerance for a, b in zip(curve, curve[1:], strict=False)
        )
        base, mid = curve[0], curve[2]
        gain = (mid - base) / base if base else 0.0
        passed = monotone and gain >= GAIN_THRESHOLD
        verdicts[name] = "PASS" if passed else "STOP"
        print(
            f"{name:<18}{base:>9.4f}{mid:>10.4f}{gain * 100:>8.1f}%"
            f"{str(monotone):>10}{verdicts[name]:>9}"
        )

    out = ROOT / "runs" / "prefix-probe.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "evidence": "[L2] parameter sensitivity, deterministic simulation",
                "capacity_gib": CAPACITY_GIB,
                "seeds": list(SEEDS),
                "h_values": list(H_VALUES),
                "workloads": list(WORKLOADS),
                "gain_threshold": GAIN_THRESHOLD,
                "verdicts": verdicts,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    passed = sum(value == "PASS" for value in verdicts.values())
    print(f"\n[L2] verdict: {passed}/{len(verdicts)} workloads pass; raw rows -> {out}")
    print("[L2] caveat: prefix_hit_ratio is an assumed upper bound, not an implemented cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
