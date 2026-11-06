from phaseweaver.policy import (
    ClusterSnapshot,
    PinAlwaysPolicy,
    TtlEvictPolicy,
    YieldOnToolPolicy,
)
from phaseweaver.types import Disposition

CAP = 8 * 1024**3


def snap(resident: int = 0, capacity: int = CAP) -> ClusterSnapshot:
    return ClusterSnapshot(
        now=0.0,
        capacity_bytes=capacity,
        resident_bytes=resident,
        queued=0,
        reasoning=1,
        slack_by_unit={"u1": 60.0},
        recent_thrash={},
    )


def test_pin_always_never_yields() -> None:
    assert PinAlwaysPolicy().disposition_for("u1", snap(), 60.0) is Disposition.RETAIN


def test_ttl_evict_uses_ttl() -> None:
    policy = TtlEvictPolicy(ttl_s=5.0)
    assert policy.disposition_for("u1", snap(), 2.0) is Disposition.RETAIN
    assert policy.disposition_for("u1", snap(), 10.0) is Disposition.DROP


def test_yield_offloads_unless_not_available() -> None:
    policy = YieldOnToolPolicy(retain_max_s=2.0, offload_max_s=20.0)
    assert policy.disposition_for("u1", snap(), 10.0) is Disposition.OFFLOAD
    policy_stdin = YieldOnToolPolicy(
        retain_max_s=3.0, offload_max_s=30.0, offload_available=False
    )
    assert policy_stdin.disposition_for("u1", snap(), 10.0) is Disposition.DROP


def test_snapshot_pressure() -> None:
    assert snap(0).pressure == 0.0
    assert snap(CAP).pressure == 1.0
