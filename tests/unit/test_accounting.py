from phaseweaver.accounting import lease_bytes, max_tokens_for, record_park, state_for
from phaseweaver.backends.base import BackendCaps
from phaseweaver.clock import VirtualClock
from phaseweaver.journal import EventJournal
from phaseweaver.leases import Budget, LeaseBroker
from phaseweaver.types import ContextState, Disposition


def caps(block_tokens: int = 16, bytes_per_token: int = 1000) -> BackendCaps:
    return BackendCaps(
        capacity_bytes=1024 * 1024,
        block_tokens=block_tokens,
        bytes_per_token=bytes_per_token,
        prefill_tok_per_s=1000,
        decode_tok_per_s=100,
        mem_bw_bps=10**7,
    )


def test_lease_bytes_rounds_to_blocks() -> None:
    assert lease_bytes(caps(), 16) == 16 * 1000
    assert lease_bytes(caps(), 17) == 32 * 1000  # one extra block
    assert lease_bytes(caps(), 1) == 16 * 1000


def test_max_tokens_for_quota() -> None:
    clock = VirtualClock()
    broker = LeaseBroker(
        journal=EventJournal(run_id="r", clock=clock), clock=clock, budget=Budget(10**9, 0.0)
    )
    broker.open_scope("s", 16 * 1000 * 4)  # room for exactly four blocks
    assert max_tokens_for(caps(), broker, "s") == 64
    assert max_tokens_for(caps(), broker, None) is None


def test_record_park_slides_the_window() -> None:
    history: dict[str, list[float]] = {}
    assert record_park(history, "u", 10.0, window_s=5.0) == 1
    assert record_park(history, "u", 12.0, window_s=5.0) == 2
    # 20s later both entries fell out of the window
    assert record_park(history, "u", 32.0, window_s=5.0) == 1


def test_state_for_covers_every_disposition() -> None:
    for disposition in Disposition:
        state = state_for(disposition)
        assert isinstance(state, ContextState)
    assert state_for(Disposition.RETAIN) is ContextState.RETAINED
    assert state_for(Disposition.RELEASE) is ContextState.RELEASED
