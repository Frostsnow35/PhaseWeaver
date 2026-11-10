"""Trace replay and the [L2] prefix-hit sensitivity knob."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.replay import (
    TraceEvent,
    format_report,
    load_trace,
    prefix_overlap,
    replay_compare,
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "examples" / "traces" / "sample.jsonl"
RUNS = ROOT / "runs"


def _tmp_trace(content: str) -> Path:
    """pytest's tmp_path is unusable in this sandbox; use the gitignored runs/ dir."""
    RUNS.mkdir(exist_ok=True)
    return RUNS / f"test-replay-{uuid.uuid4().hex}.jsonl"


def _reason(unit: str, prefix: str, prefix_tokens: int) -> TraceEvent:
    return TraceEvent(
        ts=0.0,
        unit_id=unit,
        kind="reason",
        prompt_tokens=4096,
        out_tokens=128,
        prefix_id=prefix,
        prefix_tokens=prefix_tokens,
    )


def _caps() -> BackendCaps:
    return BackendCaps(
        capacity_bytes=2 * 1024**3,
        block_tokens=16,
        bytes_per_token=57344,
        prefill_tok_per_s=3200,
        decode_tok_per_s=45,
        mem_bw_bps=12 * 1024**3,
        supports_offload=True,
    )


def _restore(ratio: float) -> tuple[int, int]:
    async def scenario() -> tuple[int, int]:
        clock = VirtualClock()
        clock.start_driver()
        try:
            backend = MockBackend(caps=_caps(), clock=clock, prefix_hit_ratio=ratio)
            handle = await backend.submit("u", "c", 4096)
            await backend.abort(handle, "test")
            result = await backend.restore(handle)
            return result.recompute_tokens, backend.available_bytes()
        finally:
            clock.stop_driver()

    return asyncio.run(scenario())


def test_sample_trace_is_loadable() -> None:
    events = load_trace(SAMPLE)
    assert len(events) == 72
    assert len({event.unit_id for event in events}) == 12
    assert prefix_overlap(events).saving_ratio is not None


def test_prefix_overlap_counts_blocks_not_events() -> None:
    events = [_reason(f"u{i}", "sys", 3072) for i in range(3)]
    overlap = prefix_overlap(events)
    assert overlap.logical_blocks == 3 * 192
    assert overlap.distinct_blocks == 192
    assert overlap.saving_ratio == pytest.approx(2 / 3)


def test_prefix_overlap_is_unknown_without_metadata() -> None:
    events = [TraceEvent(ts=0.0, unit_id="u", kind="reason", prompt_tokens=4096)]
    assert prefix_overlap(events).saving_ratio is None


def test_load_trace_rejects_bad_kind() -> None:
    bad = _tmp_trace("")
    bad.write_text('{"unit_id":"u","kind":"fly"}\n', encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="kind"):
            load_trace(bad)
    finally:
        bad.unlink(missing_ok=True)


def test_load_trace_rejects_invalid_json() -> None:
    bad = _tmp_trace("")
    bad.write_text("not-json\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="invalid JSON"):
            load_trace(bad)
    finally:
        bad.unlink(missing_ok=True)


def test_load_trace_rejects_empty_file() -> None:
    empty = _tmp_trace("")
    empty.write_text("# comment only\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="empty trace"):
            load_trace(empty)
    finally:
        empty.unlink(missing_ok=True)


def test_prefix_hit_zero_preserves_original_model() -> None:
    assert _restore(0.0)[0] == 4096


def test_prefix_hit_discounts_prefill_work_not_allocation() -> None:
    recompute, available = _restore(0.5)
    assert recompute == 2048
    assert available == _restore(0.0)[1]


def test_replay_compare_is_deterministic_and_labelled() -> None:
    events = load_trace(SAMPLE)
    first = replay_compare(events, seeds=(0,), policies=("B0", "B2"))
    second = replay_compare(events, seeds=(0,), policies=("B0", "B2"))
    assert first.policies == second.policies
    assert set(first.policies) == {"B0", "B2"}
    assert "[L1]" in first.evidence
    report = format_report(first)
    assert "seeds=0" in report
    assert "C=2.0GiB" in report
