"""R6: phase alignment policy.

The experiment verdict is negative (see docs/notes/experiments.md §9), but the
policy and the admission gate it relies on must still behave as specified.
"""

from __future__ import annotations

from phaseweaver.alignment import AlignedPolicy
from phaseweaver.app import build_policy
from phaseweaver.policy import ClusterSnapshot


def snap(reasoning: int, pressure: float = 0.5) -> ClusterSnapshot:
    return ClusterSnapshot(
        now=0.0,
        capacity_bytes=1000,
        resident_bytes=int(pressure * 1000),
        queued=0,
        reasoning=reasoning,
        slack_by_unit={},
        recent_thrash={},
    )


def test_align_admits_below_target_only() -> None:
    policy = AlignedPolicy(target_streams=2)
    assert policy.admit("u", snap(reasoning=0))
    assert policy.admit("u", snap(reasoning=1))
    assert not policy.admit("u", snap(reasoning=2))
    assert not policy.admit("u", snap(reasoning=5))


def test_align_never_admits_under_pressure() -> None:
    policy = AlignedPolicy(target_streams=4)
    assert not policy.admit("u", snap(reasoning=0, pressure=0.99))


def test_align_is_addressable_by_name() -> None:
    policy = build_policy("align3")
    assert isinstance(policy, AlignedPolicy)
    assert policy.target_streams == 3


def test_align_inherits_b2_dispositions() -> None:
    aligned = AlignedPolicy(target_streams=2)
    plain = build_policy("B2")
    from phaseweaver.types import Disposition

    for unit_slack in (0.5, 10.0, 600.0):
        a = aligned.disposition_for("u1", snap(reasoning=1), unit_slack)
        b = plain.disposition_for("u1", snap(reasoning=1), unit_slack)
        assert isinstance(a, Disposition)
        assert a == b
