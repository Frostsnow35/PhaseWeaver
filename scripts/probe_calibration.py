"""[L2] Re-run breakthrough A's two gates with real-GPU calibrated parameters.

Input is the ``calibration.json`` produced by ``_local/kaggle_l3_calibration.ipynb``.
The notebook only measures micro-benchmark constants; this script decides whether
those constants change the *decision* about implementing prefix-block refcounting.

Two gates, both from scripts/probe_prefix.py and scripts/probe_capacity.py:

  prefix   B2 goodput gain from prefix_hit_ratio 0 -> 0.5 (needs >= 5% per workload)
  capacity B2 goodput gain from effective capacity C -> 1.5*C (needs >= 5%)

A reopens only when >= 4/6 workloads pass BOTH gates. Every variant is a
deterministic simulation with 3 seeds; raw rows go to runs/calibration-probe.json
(gitignored). Nothing here is an [L3] end-to-end performance claim.

    python scripts/probe_calibration.py --calibration runs/kaggle-l3/calibration.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import replace
from pathlib import Path

from phaseweaver.backends.base import BackendCaps
from phaseweaver.demo import Workload, caps, run_one

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = (
    "w2.toml",
    "w3-queue.toml",
    "w5-fanout.toml",
    "w8-handoff.toml",
    "w9-fanout.toml",
    "w10-nested.toml",
)
SEEDS = (0, 1, 2)
CAPACITY_GIB = 2.0
GAIN_THRESHOLD = 0.05
PASS_COUNT = 4
BLOCK_TOKENS = 16
BYTES_PER_TOKEN = 57344


def variant_caps(prefill: int, decode: int, mem_bw: int, alpha: float) -> BackendCaps:
    return replace(
        caps(CAPACITY_GIB),
        bytes_per_token=BYTES_PER_TOKEN,
        prefill_tok_per_s=prefill,
        decode_tok_per_s=decode,
        mem_bw_bps=mem_bw,
        contention_alpha=alpha,
    )


def goodput(name: str, seed: int, policy: str, *, capacity_gib: float,
            hit: float, backend_caps: BackendCaps) -> float:
    workload = Workload.load(ROOT / "examples" / "workloads" / name)
    workload.seed = seed
    workload.prefix_hit_ratio = hit
    report = asyncio.run(
        run_one(
            workload,
            policy,
            capacity_gib=capacity_gib,
            backend_caps=replace(
                backend_caps, capacity_bytes=int(capacity_gib * 1024**3)
            ),
        )
    )
    return float(report.summary_row()["goodput"])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def gate_pair(backend_caps: BackendCaps) -> tuple[list[float], list[float]]:
    """Return (prefix gains, capacity gains), one value per workload."""
    prefix_gains: list[float] = []
    capacity_gains: list[float] = []
    for name in WORKLOADS:
        def gp(workload_name: str, hit: float, capacity: float) -> float:
            return mean([
                goodput(workload_name, seed, "B2", capacity_gib=capacity, hit=hit,
                        backend_caps=backend_caps)
                for seed in SEEDS
            ])

        base = gp(name, 0.0, CAPACITY_GIB)
        prefix_gains.append((gp(name, 0.5, CAPACITY_GIB) - base) / base if base else 0.0)
        capacity_gains.append(
            (gp(name, 0.0, CAPACITY_GIB * 1.5) - base) / base if base else 0.0
        )
    return prefix_gains, capacity_gains


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration", type=Path,
        default=ROOT / "runs" / "kaggle-l3" / "calibration.json",
    )
    parser.add_argument(
        "--rates", type=Path, default=ROOT / "runs" / "kaggle-l3" / "rates.json",
        help="per-prompt-length prefill rates from notebook Cell 2",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "calibration-probe.json")
    args = parser.parse_args()

    cal = json.loads(args.calibration.read_text(encoding="utf-8"))
    reported = cal.get("prefill_tok_per_s_measured")
    decode = cal.get("decode_tok_per_s_measured") or 45
    mem_bw = cal.get("mem_bw_bps_measured") or int(12 * 1024**3)
    alpha_measured = cal.get("contention_alpha_measured")
    if alpha_measured is None:
        # v1 notebooks only stored the raw batch curve; fit the same model here.
        curve_alpha = cal.get("alpha_curve", [])
        if len(curve_alpha) > 1:
            base = float(curve_alpha[0]["tok_per_s_per_stream"])
            fits = [
                (base / float(r["tok_per_s_per_stream"]) - 1) / (int(r["batch"]) - 1)
                for r in curve_alpha[1:]
            ]
            alpha_measured = round(statistics.median(fits), 4)
    alpha_measured = alpha_measured or 0.15
    rates_rows: list[dict[str, object]] = (
        json.loads(args.rates.read_text(encoding="utf-8")) if args.rates.exists() else []
    )
    # v2 emits a per-length curve and deliberately no scalar. For a sensitivity
    # variant only, fold the curve into a mean; never write it to BackendCaps.
    curve = list(cal.get("prefill_rate_by_prompt_tokens", {}).values())
    rates = [float(v) for v in curve] or [
        float(r["prefill_tok_per_s"]) for r in rates_rows
        if float(r.get("prefill_tok_per_s", 0)) >= 1000
    ]
    fast, slow = (max(rates), min(rates)) if rates else (7109.0, 2562.0)
    reported = reported or (sum(rates) / len(rates) if rates else 3200)

    variants: list[tuple[str, BackendCaps]] = [
        ("mock defaults", variant_caps(3200, 45, int(12 * 1024**3), 0.15)),
        ("L3 as reported", variant_caps(int(reported), int(decode), int(mem_bw), 0.15)),
        ("L3 decode+bandwidth", variant_caps(3200, int(decode), int(mem_bw), 0.15)),
        ("L3 fast prefill", variant_caps(int(fast), int(decode), int(mem_bw), 0.15)),
        ("L3 slow prefill", variant_caps(int(slow), int(decode), int(mem_bw), 0.15)),
        ("L3 + measured alpha", variant_caps(3200, int(decode), int(mem_bw),
                                             float(alpha_measured))),
    ]

    print(
        f"[L2] calibration robustness  C={CAPACITY_GIB}GiB  seeds={','.join(map(str, SEEDS))}"
        f"  thresholds: gain>={GAIN_THRESHOLD:.0%}, >={PASS_COUNT}/6 workloads"
    )
    print(f"[L2] source: {args.calibration}")
    print(
        f"{'variant':<22}{'prefill':>9}{'decode':>8}{'alpha':>7}"
        f"{'prefix pass':>13}{'prefix mean':>13}{'cap pass':>10}{'cap mean':>10}{'A':>8}"
    )
    rows: list[dict[str, object]] = []
    for label, backend_caps in variants:
        prefix_gains, capacity_gains = gate_pair(backend_caps)
        p_pass = sum(g >= GAIN_THRESHOLD for g in prefix_gains)
        c_pass = sum(g >= GAIN_THRESHOLD for g in capacity_gains)
        reopen = p_pass >= PASS_COUNT and c_pass >= PASS_COUNT
        print(
            f"{label:<22}{backend_caps.prefill_tok_per_s:>9}"
            f"{backend_caps.decode_tok_per_s:>8}{backend_caps.contention_alpha:>7.2f}"
            f"{str(p_pass) + '/6':>13}{mean(prefix_gains) * 100:>12.1f}%"
            f"{str(c_pass) + '/6':>10}{mean(capacity_gains) * 100:>9.1f}%"
            f"{('REOPEN' if reopen else 'STOP'):>8}"
        )
        rows.append({
            "variant": label,
            "prefill_tok_per_s": backend_caps.prefill_tok_per_s,
            "decode_tok_per_s": backend_caps.decode_tok_per_s,
            "mem_bw_bps": backend_caps.mem_bw_bps,
            "contention_alpha": backend_caps.contention_alpha,
            "prefix_gains": prefix_gains,
            "capacity_gains": capacity_gains,
            "prefix_pass": p_pass,
            "capacity_pass": c_pass,
            "verdict": "REOPEN" if reopen else "STOP",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "evidence": "[L2] deterministic simulation with [L3] micro-benchmark constants",
                "caveat": "calibration constants come from a 0.5B HF eager run; "
                          "bytes_per_token is 7B. Not a same-model measurement.",
                "calibration": str(args.calibration),
                "capacity_gib": CAPACITY_GIB,
                "seeds": list(SEEDS),
                "gain_threshold": GAIN_THRESHOLD,
                "pass_count": PASS_COUNT,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    n = sum(r["verdict"] == "REOPEN" for r in rows)
    print(f"\n[L2] {n}/{len(rows)} variants would reopen A; raw rows -> {args.out}")
    print("[L2] prefix mean is skewed by w3-queue; the pass count is the decision criterion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
