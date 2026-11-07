"""Regression tests for audit findings F1 (fork bookkeeping leak) and F6 (error taxonomy)."""

from __future__ import annotations

import asyncio

import pytest

from phaseweaver.app import mock_caps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.leases import Budget
from phaseweaver.policy import PinAlwaysPolicy
from phaseweaver.runtime import AgentHandle, Runtime, RuntimeConfig


def build(factory=None, **config) -> Runtime:  # type: ignore[no-untyped-def]
    clock = VirtualClock()
    clock.start_driver()
    caps = mock_caps(2.0)
    backend = factory(clock, caps) if factory else MockBackend(caps=caps, clock=clock)
    return Runtime(
        backend=backend,
        clock=clock,
        budget=Budget(total_bytes=caps.capacity_bytes, headroom_frac=0.05),
        policy=PinAlwaysPolicy(),
        config=RuntimeConfig(run_id="faults", **config),
    )


class _HandoffBoom(MockBackend):
    async def handoff(self, handle, new_unit_id, new_context_id):  # type: ignore[no-untyped-def]
        raise RuntimeError("handoff exploded")


class _CompleteBoom(MockBackend):
    async def complete(self, handle, out_tokens):  # type: ignore[no-untyped-def]
        raise RuntimeError("engine exploded")


async def _child(handle: AgentHandle) -> None:
    await handle.reason(prompt_tokens=256, out_tokens=16)


def test_handoff_failure_does_not_leak_fork_depth() -> None:
    """F1: an exception during inheritance must restore depth and close the scope."""

    async def scenario() -> Runtime:
        rt = build(lambda clock, caps: _HandoffBoom(caps=caps, clock=clock))
        await rt.start()

        async def parent(handle: AgentHandle) -> None:
            await handle.reason(prompt_tokens=512, out_tokens=16)
            with pytest.raises(RuntimeError):
                await handle.fork("child", _child, inherit="handoff")
            # The parent keeps working; the next fork must see a clean slate.
            await handle.reason(prompt_tokens=520, out_tokens=16)

        outcome = await rt.run_task("t", parent)
        assert outcome.phase == "COMPLETED"
        parent_unit = next(u for u in rt.registry.all_units() if u.kind.value == "agent")
        assert rt._fork_depth.get(parent_unit.unit_id, 0) == 0
        await rt.shutdown()
        return rt

    asyncio.run(scenario())


def test_unexpected_error_recorded_with_traceback() -> None:
    async def scenario() -> Runtime:
        def err_factory(clock, caps):  # type: ignore[no-untyped-def]
            return _CompleteBoom(caps=caps, clock=clock)
        rt = build(err_factory)
        await rt.start()
        outcome = await rt.run_task("t", _child)
        assert outcome.phase == "FAILED"
        assert outcome.error is not None and "RuntimeError" in outcome.error
        errors = [e for e in rt.journal.events() if e.event == "run_error"]
        assert errors, "unexpected errors must be journaled for post-mortem"
        assert "Traceback" in errors[0].payload["traceback"]
        assert not rt.broker.active_leases()
        await rt.shutdown()
        return rt

    asyncio.run(scenario())


def test_strict_errors_propagate_after_cleanup() -> None:
    async def scenario() -> None:
        def err_factory(clock, caps):  # type: ignore[no-untyped-def]
            return _CompleteBoom(caps=caps, clock=clock)
        rt = build(err_factory, strict_errors=True)
        await rt.start()
        with pytest.raises(RuntimeError, match="engine exploded"):
            await rt.run_task("t", _child)
        # cleanup still ran before the exception escaped
        assert not rt.broker.active_leases()
        await rt.shutdown()

    asyncio.run(scenario())
