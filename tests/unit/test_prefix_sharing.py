"""KV fork: the child must reference the parent's prefix instead of copying it."""

from phaseweaver.clock import VirtualClock
from phaseweaver.journal import EventJournal
from phaseweaver.leases import Budget, LeaseBroker
from phaseweaver.prefix import PrefixRegistry
from phaseweaver.types import LeaseKind


def test_registry_refcounts_and_frees() -> None:
    reg = PrefixRegistry()
    reg.acquire("p1", 100, 1000, "a")
    reg.acquire("p1", 100, 1000, "b")
    assert reg.unique_bytes() == 1000
    assert reg.get("p1") is not None
    assert reg.get("p1").refs == 2  # one per owner
    assert reg.release("p1", "a") is False
    assert reg.release("p1", "b") is True
    assert reg.get("p1") is None
    assert reg.unique_bytes() == 0


def test_prefix_is_charged_once() -> None:
    clock = VirtualClock()
    journal = EventJournal(run_id="r", clock=clock)
    broker = LeaseBroker(journal=journal, clock=clock, budget=Budget(10_000, 0.0))

    parent = broker.acquire(
        "u1", "c1", LeaseKind.KV_BYTES, 4000, 5, prefix_key="p1", shared_bytes=0
    )
    assert parent is not None
    assert broker.used(LeaseKind.KV_BYTES) == 4000

    # Child reuses all 4000 bytes: nominal amount 4000, charged 0.
    child = broker.acquire(
        "u2", "c2", LeaseKind.KV_BYTES, 4000, 5, prefix_key="p1", shared_bytes=4000
    )
    assert child is not None
    assert child.amount == 4000
    assert child.charged == 0
    assert broker.used(LeaseKind.KV_BYTES) == 4000  # not 8000

    # A child with a larger context pays only for the delta.
    big = broker.acquire(
        "u3", "c3", LeaseKind.KV_BYTES, 5000, 5, prefix_key="p1", shared_bytes=4000
    )
    assert big is not None
    assert big.charged == 1000
    assert broker.used(LeaseKind.KV_BYTES) == 5000

    broker.release(parent.lease_id)
    assert broker.used(LeaseKind.KV_BYTES) == 5000  # group still referenced
    broker.release(child.lease_id)
    broker.release(big.lease_id)
    assert broker.used(LeaseKind.KV_BYTES) == 0
    assert broker.prefixes.unique_bytes() == 0
