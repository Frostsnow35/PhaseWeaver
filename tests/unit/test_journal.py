import asyncio
import json
import uuid
from pathlib import Path

from phaseweaver.clock import VirtualClock
from phaseweaver.journal import EventJournal

RUNS = Path("runs")


def tmp_file() -> Path:
    RUNS.mkdir(exist_ok=True)
    return RUNS / f"test-{uuid.uuid4().hex}.jsonl"


def test_append_assigns_monotonic_seq() -> None:
    clock = VirtualClock()
    j = EventJournal(run_id="r1", clock=clock)
    e1 = j.append("run_started", payload={"x": 1})
    e2 = j.append("unit_created", unit_id="r1/root")
    assert (e1.seq, e2.seq) == (1, 2)
    assert e2.unit_id == "r1/root"


def test_jsonl_file_is_valid() -> None:
    path = tmp_file()
    try:
        j = EventJournal(run_id="r1", clock=VirtualClock(), path=path)
        j.append("run_started")
        j.append("run_finished", payload={"ok": True})
        j.close()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["payload"] == {"ok": True}
    finally:
        path.unlink(missing_ok=True)


def test_digest_is_stable_for_same_input() -> None:
    def build() -> EventJournal:
        j = EventJournal(run_id="r1", clock=VirtualClock())
        j.append("run_started")
        j.append("phase_transitioned", unit_id="u", phase_from="QUEUED", phase_to="REASONING")
        return j

    assert build().digest() == build().digest()


def test_virtual_clock_ordering() -> None:
    async def scenario() -> list[float]:
        clock = VirtualClock()
        clock.start_driver()
        marks: list[float] = []

        async def worker(dt: float) -> None:
            await clock.wait(dt)
            marks.append(clock.now())

        try:
            await asyncio.gather(worker(2.0), worker(1.0), worker(3.0))
        finally:
            clock.stop_driver()
        return marks

    assert asyncio.run(scenario()) == [1.0, 2.0, 3.0]
