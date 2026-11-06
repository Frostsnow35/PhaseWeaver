from phaseweaver.clock import VirtualClock
from phaseweaver.journal import EventJournal
from phaseweaver.leases import Budget, LeaseBroker
from phaseweaver.types import LeaseKind


def make_broker(total: int = 1000) -> LeaseBroker:
    clock = VirtualClock()
    return LeaseBroker(
        journal=EventJournal(run_id="r1", clock=clock),
        clock=clock,
        budget=Budget(total_bytes=total, headroom_frac=0.0),
    )


def test_acquire_within_budget() -> None:
    broker = make_broker()
    lease = broker.acquire("u1", "c1", LeaseKind.KV_BYTES, 600, priority=5)
    assert lease is not None
    assert broker.used(LeaseKind.KV_BYTES) == 600


def test_acquire_over_budget_returns_none() -> None:
    broker = make_broker()
    assert broker.acquire("u1", "c1", LeaseKind.KV_BYTES, 600, priority=5) is not None
    assert broker.acquire("u2", "c2", LeaseKind.KV_BYTES, 600, priority=5) is None
    assert broker.used(LeaseKind.KV_BYTES) == 600


def test_release_frees_capacity() -> None:
    broker = make_broker()
    lease = broker.acquire("u1", "c1", LeaseKind.KV_BYTES, 1000, priority=5)
    assert lease is not None
    broker.release(lease.lease_id)
    assert broker.used(LeaseKind.KV_BYTES) == 0
    assert broker.acquire("u2", "c2", LeaseKind.KV_BYTES, 1000, priority=5) is not None


def test_headroom_is_reserved() -> None:
    clock = VirtualClock()
    broker = LeaseBroker(
        journal=EventJournal(run_id="r1", clock=clock),
        clock=clock,
        budget=Budget(total_bytes=1000, headroom_frac=0.1),
    )
    assert broker.available(LeaseKind.KV_BYTES) == 900


def test_release_all_is_forced() -> None:
    broker = make_broker()
    broker.acquire("u1", "c1", LeaseKind.KV_BYTES, 100, priority=5)
    broker.acquire("u2", "c2", LeaseKind.KV_BYTES, 100, priority=5)
    broker.release_all()
    assert broker.used(LeaseKind.KV_BYTES) == 0


def test_scope_quota_caps_isolated_child() -> None:
    broker = make_broker(10_000)
    broker.open_scope("iso:1", 300)
    assert broker.acquire("u1", "c1", LeaseKind.KV_BYTES, 300, 5, scope="iso:1") is not None
    # over the scope limit even though global budget has room
    assert broker.acquire("u1", "c1", LeaseKind.KV_BYTES, 100, 5, scope="iso:1") is None
    assert broker.scope_usage("iso:1") == 300
    broker.close_scope("iso:1")
    assert broker.scope_limit("iso:1") is None


def test_shared_scope_has_no_cap() -> None:
    broker = make_broker(10_000)
    assert broker.acquire("u1", "c1", LeaseKind.KV_BYTES, 900, 5, scope=None) is not None
