from phaseweaver.autopilot import AutopilotConfig, AutopilotPolicy
from phaseweaver.policy import ClusterSnapshot
from phaseweaver.types import Disposition


def snap(
    *, resident: int, capacity: int = 1000, queued: int = 0, slack: float = 60.0
) -> ClusterSnapshot:
    return ClusterSnapshot(
        now=100.0,
        capacity_bytes=capacity,
        resident_bytes=resident,
        queued=queued,
        reasoning=1,
        slack_by_unit={"u1": slack},
        recent_thrash={},
    )


def test_hot_pressure_tightens_retention() -> None:
    policy = AutopilotPolicy()
    for _ in range(6):
        policy.disposition_for("u1", snap(resident=950, queued=3, slack=600.0), 30.0)
    assert policy.config.retain_max_s == AutopilotConfig().retain_when_hot_s


def test_cool_system_keeps_contexts_warm() -> None:
    policy = AutopilotPolicy()
    for _ in range(6):
        policy.disposition_for("u1", snap(resident=100, queued=0, slack=600.0), 30.0)
    assert policy.config.retain_max_s == AutopilotConfig().retain_when_cool_s


def test_urgent_unit_is_never_dropped() -> None:
    policy = AutopilotPolicy()
    # slack below the tool duration: the unit cannot afford to lose its context
    assert (
        policy.disposition_for("u1", snap(resident=950, queued=5, slack=0.5), 5.0)
        is Disposition.RETAIN
    )
