from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from phaseweaver import PhaseWeaver, ToolSpec
from phaseweaver.timeline import build_timeline
from phaseweaver.trace import format_trace, trace_unit

RUNS = Path("runs")


def workdir() -> Path:
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"obs-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _agent(unit) -> None:  # type: ignore[no-untyped-def]
    await unit.reason(prompt_tokens=512, out_tokens=32)
    await unit.call_tool(ToolSpec("slow", "python", expected_duration_s=8.0), {"code": "pass"})
    await unit.reason(prompt_tokens=600, out_tokens=32)


def run() -> PhaseWeaver:
    async def scenario() -> PhaseWeaver:
        async with PhaseWeaver(profile="research", capacity_gib=2.0) as pw:
            await pw.run("t", _agent)
            return pw

    return asyncio.run(scenario())


def test_timeline_covers_the_run() -> None:
    pw = run()
    timeline = build_timeline(pw.runtime.journal)
    assert timeline.wall_clock_s > 0
    agent = next(u for u in pw.runtime.registry.all_units() if u.kind.value == "agent")
    names = {
        span.name
        for span in timeline.spans
        if span.unit_id == agent.unit_id and span.kind == "phase"
    }
    assert {"QUEUED", "REASONING", "ACTING"} & names
    assert timeline.total(agent.unit_id, "ACTING") == 8.0  # simulated tool duration


def test_timeline_exports_jsonl() -> None:
    pw = run()
    out = workdir() / "timeline.jsonl"
    build_timeline(pw.runtime.journal).to_jsonl(out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines and "unit_id" in lines[0]
    out.unlink(missing_ok=True)


def test_trace_attributes_time_and_explains() -> None:
    pw = run()
    unit = next(u for u in pw.runtime.registry.all_units() if u.kind.value == "agent")
    trace = trace_unit(pw.runtime.journal, unit.unit_id)
    assert trace.latency_s > 8.0
    assert trace.phases.get("ACTING", 0) == 8.0
    assert trace.park_count >= 1
    assert "工具" in trace.hint()
    text = format_trace(trace)
    assert "diagnosis" in text and "timeline" in text


def test_journal_roundtrip() -> None:
    from phaseweaver.journal import EventJournal

    pw = run()
    path = workdir() / "roundtrip.jsonl"
    journal = EventJournal(run_id="rt", clock=pw.runtime.clock, path=path)
    for event in pw.runtime.journal.events():
        journal.append(
            event.event,
            unit_id=event.unit_id,
            phase_from=event.phase_from,
            phase_to=event.phase_to,
            cause_seq=event.cause_seq,
            payload=event.payload,
        )
    journal.close()
    loaded = EventJournal.load(path)
    assert loaded.digest() != ""
    assert len(list(loaded.events())) > 0
    path.unlink(missing_ok=True)
