"""Runtime lifecycle contract: using a closed runtime must fail loudly, not silently."""

from __future__ import annotations

import asyncio

import pytest

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.errors import RuntimeClosed
from phaseweaver.leases import Budget
from phaseweaver.policy import PinAlwaysPolicy
from phaseweaver.runtime import AgentHandle, Runtime, RuntimeConfig


def caps() -> BackendCaps:
    return BackendCaps(
        capacity_bytes=1024 * 1024,
        block_tokens=16,
        bytes_per_token=1000,
        prefill_tok_per_s=1000,
        decode_tok_per_s=100,
        mem_bw_bps=10**7,
    )


async def _agent(handle: AgentHandle) -> None:
    await handle.reason(prompt_tokens=64, out_tokens=8)


def build() -> Runtime:
    clock = VirtualClock()
    c = caps()
    return Runtime(
        backend=MockBackend(caps=c, clock=clock),
        clock=clock,
        budget=Budget(total_bytes=c.capacity_bytes, headroom_frac=0.0),
        policy=PinAlwaysPolicy(),
        config=RuntimeConfig(run_id="closed"),
    )


def test_run_before_start_is_rejected() -> None:
    async def scenario() -> None:
        rt = build()
        with pytest.raises(RuntimeClosed):
            await rt.run_task("t", _agent)

    asyncio.run(scenario())


def test_run_after_shutdown_is_rejected() -> None:
    async def scenario() -> None:
        rt = build()
        assert isinstance(rt.clock, VirtualClock)
        rt.clock.start_driver()
        await rt.start()
        await rt.run_task("t", _agent)
        await rt.shutdown()
        with pytest.raises(RuntimeClosed):
            await rt.run_task("t2", _agent)
        rt.clock.stop_driver()

    asyncio.run(scenario())


def test_shutdown_is_idempotent() -> None:
    async def scenario() -> None:
        rt = build()
        assert isinstance(rt.clock, VirtualClock)
        rt.clock.start_driver()
        await rt.start()
        await rt.run_task("t", _agent)
        await rt.shutdown()
        await rt.shutdown()  # must not raise
        rt.clock.stop_driver()

    asyncio.run(scenario())


class _StalledBackend(MockBackend):
    """An engine that accepts work and never returns - a hung driver."""

    async def complete(self, handle, out_tokens):  # type: ignore[no-untyped-def]
        await asyncio.sleep(30)
        return await super().complete(handle, out_tokens)


def test_stalled_backend_fails_the_unit_not_the_tree() -> None:
    async def scenario() -> None:
        c = caps()
        backend = _StalledBackend(caps=c, clock=VirtualClock())
        rt = Runtime(
            backend=backend,
            clock=VirtualClock(),
            budget=Budget(total_bytes=c.capacity_bytes, headroom_frac=0.0),
            policy=PinAlwaysPolicy(),
            config=RuntimeConfig(run_id="stall", backend_timeout_s=0.05),
        )
        await rt.start()
        outcome = await rt.run_task("t", _agent)
        await rt.shutdown()
        assert outcome.phase == "FAILED"
        assert outcome.error is not None and "timed out" in outcome.error
        assert not rt.broker.active_leases()

    asyncio.run(scenario())
