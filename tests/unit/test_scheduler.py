from phaseweaver.policy import ClusterSnapshot
from phaseweaver.scheduler import SchedulerConfig, SystemScheduler
from phaseweaver.types import Disposition


def snap(
    *, resident: int, capacity: int = 1000, slack: float = 60.0, thrash: int = 0
) -> ClusterSnapshot:
    return ClusterSnapshot(
        now=100.0,
        capacity_bytes=capacity,
        resident_bytes=resident,
        queued=1,
        reasoning=1,
        slack_by_unit={"u1": slack},
        recent_thrash={"u1": thrash},
    )


def test_urgent_unit_is_retained() -> None:
    sched = SystemScheduler(SchedulerConfig())
    assert sched.disposition_for("u1", snap(resident=100, slack=0.5), 1.0) is Disposition.RETAIN


def test_relaxed_unit_yields_even_for_short_tool() -> None:
    sched = SystemScheduler(SchedulerConfig())
    assert sched.disposition_for("u1", snap(resident=100, slack=600.0), 1.0) is Disposition.OFFLOAD


def test_hopeless_unit_is_not_prioritized() -> None:
    sched = SystemScheduler(SchedulerConfig())
    assert sched.disposition_for("u1", snap(resident=100, slack=-30.0), 1.0) is Disposition.DROP


def test_high_pressure_degrades_one_level() -> None:
    sched = SystemScheduler(SchedulerConfig())
    assert sched.disposition_for("u1", snap(resident=950, slack=600.0), 1.0) is Disposition.DROP


def test_thrashing_context_is_promoted() -> None:
    sched = SystemScheduler(SchedulerConfig())
    sched._state("u1").park_count_window = 3
    assert (
        sched.disposition_for("u1", snap(resident=100, slack=600.0), 30.0) is Disposition.RETAIN
    )


def test_cooldown_suppresses_repeated_park() -> None:
    sched = SystemScheduler(SchedulerConfig(park_cooldown_s=10.0))
    sched.record_park("u1", now=95.0)
    assert (
        sched.disposition_for("u1", snap(resident=100, slack=600.0), 30.0) is Disposition.RETAIN
    )
