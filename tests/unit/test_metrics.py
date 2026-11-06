from phaseweaver.clock import VirtualClock
from phaseweaver.journal import EventJournal
from phaseweaver.metrics import MetricsReducer, percentile


def build_journal() -> EventJournal:
    clock = VirtualClock()
    j = EventJournal(run_id="r1", clock=clock)
    j.append("run_started")
    j.append(
        "unit_created",
        unit_id="r1/root",
        payload={"kind": "agent", "arrival_ts": 0.0, "deadline": 100.0},
    )
    j.append("phase_transitioned", unit_id="r1/root", payload={"from": "PENDING", "to": "QUEUED"})
    j.append("phase_transitioned", unit_id="r1/root", payload={"from": "QUEUED", "to": "REASONING"})
    j.append(
        "context_state_changed",
        unit_id="r1/root",
        payload={"state": "ACTIVE", "resident_bytes": 1024},
    )
    j.append(
        "phase_transitioned",
        unit_id="r1/root",
        payload={"from": "REASONING", "to": "COMPLETED"},
    )
    return j


def test_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 99) == 7.0


def test_report_counts_completed_task() -> None:
    report = MetricsReducer.replay(build_journal())
    assert report.completed == 1
    assert report.slo_attainment_rate == 1.0
    assert len(report.latencies_ms) == 1
    assert report.peak_resident_bytes == 1024


def test_tool_calls_are_not_counted_as_tasks() -> None:
    j = build_journal()
    j.append(
        "unit_created",
        unit_id="r1/root/tool-0",
        payload={"kind": "tool_call", "arrival_ts": 0.0, "deadline": 100.0},
    )
    j.append(
        "phase_transitioned",
        unit_id="r1/root/tool-0",
        payload={"from": "ACTING", "to": "COMPLETED"},
    )
    report = MetricsReducer.replay(j)
    assert report.completed == 1


def test_deadline_is_absolute_not_a_duration() -> None:
    """Regression: comparing latency against an absolute deadline always passed."""

    def run(finish_at: float) -> float:
        # A clock whose reads return the scripted timeline.
        class ScriptedClock:
            def __init__(self) -> None:
                self.t = 0.0

            def now(self) -> float:
                return self.t

            async def wait(self, dt: float) -> None:
                self.t += dt

        sc = ScriptedClock()
        j = EventJournal(run_id="r1", clock=sc)
        j.append(
            "unit_created",
            unit_id="r1/root",
            payload={"kind": "agent", "arrival_ts": 10.0, "deadline": 30.0},
        )
        j.append(
            "phase_transitioned",
            unit_id="r1/root",
            payload={"from": "QUEUED", "to": "REASONING"},
        )
        sc.t = finish_at
        j.append(
            "phase_transitioned",
            unit_id="r1/root",
            payload={"from": "REASONING", "to": "COMPLETED"},
        )
        return MetricsReducer.replay(j).slo_attainment_rate

    assert run(25.0) == 1.0  # finished before absolute deadline 30
    assert run(35.0) == 0.0  # finished after it
