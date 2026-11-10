"""Calibration hook on run_one: no behaviour change by default, effective when set."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from phaseweaver.demo import Workload, caps, run_one

ROOT = Path(__file__).resolve().parents[2]
W2 = ROOT / "examples" / "workloads" / "w2.toml"


def _workload() -> Workload:
    workload = Workload.load(W2)
    workload.seed = 0
    return workload


def test_default_caps_are_byte_identical_to_explicit_caps() -> None:
    default = asyncio.run(run_one(_workload(), "B2", capacity_gib=2.0))
    explicit = asyncio.run(
        run_one(_workload(), "B2", capacity_gib=2.0, backend_caps=caps(2.0))
    )
    assert default.summary_row() == explicit.summary_row()


def test_slower_decode_rate_changes_the_result() -> None:
    baseline = asyncio.run(run_one(_workload(), "B2", capacity_gib=2.0))
    slower = asyncio.run(
        run_one(
            _workload(),
            "B2",
            capacity_gib=2.0,
            backend_caps=replace(caps(2.0), decode_tok_per_s=10),
        )
    )
    assert slower.goodput < baseline.goodput
    assert slower.latency_p99_ms > baseline.latency_p99_ms
