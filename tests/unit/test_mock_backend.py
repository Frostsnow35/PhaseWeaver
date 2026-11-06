import asyncio

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.journal import EventJournal
from phaseweaver.types import Disposition


def caps() -> BackendCaps:
    return BackendCaps(
        capacity_bytes=1024 * 1024,
        block_tokens=16,
        bytes_per_token=1000,
        prefill_tok_per_s=1000,
        decode_tok_per_s=100,
        mem_bw_bps=10**7,
        supports_offload=True,
    )


def make() -> MockBackend:
    clock = VirtualClock()
    clock.start_driver()
    journal = EventJournal(run_id="r1", clock=clock)
    return MockBackend(caps=caps(), clock=clock, journal=journal)


def test_submit_allocates_blocks() -> None:
    async def scenario() -> None:
        backend = make()
        handle = await backend.submit("u1", "c1", 32)
        assert handle.resident_tokens == 32
        assert backend.available_bytes() == 1024 * 1024 - 32 * 1000

    asyncio.run(scenario())


def test_park_drop_frees_and_restore_recomputes() -> None:
    async def scenario() -> None:
        backend = make()
        handle = await backend.submit("u1", "c1", 64)
        park = await backend.park(handle, Disposition.DROP)
        assert park.applied is Disposition.DROP
        assert backend.available_bytes() == 1024 * 1024
        restore = await backend.restore(handle)
        assert restore.recompute_tokens == 64
        assert backend.available_bytes() == 1024 * 1024 - 64 * 1000

    asyncio.run(scenario())


def test_park_retain_keeps_residency() -> None:
    async def scenario() -> None:
        backend = make()
        handle = await backend.submit("u1", "c1", 64)
        before = backend.available_bytes()
        park = await backend.park(handle, Disposition.RETAIN)
        assert park.applied is Disposition.RETAIN
        assert backend.available_bytes() == before

    asyncio.run(scenario())


def test_offload_moves_bytes_with_cost() -> None:
    async def scenario() -> None:
        backend = make()
        handle = await backend.submit("u1", "c1", 64)
        park = await backend.park(handle, Disposition.OFFLOAD)
        assert park.applied is Disposition.OFFLOAD
        assert park.moved_bytes == 64 * 1000
        assert backend.available_bytes() == 1024 * 1024
        events = [e.kind.value for e in backend.engine_events()]
        assert "block_removed" in events

    asyncio.run(scenario())
