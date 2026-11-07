"""Failure-injection suite.

These are the paths that only bite in production: tools that fail, tasks that are
cancelled mid-flight, and the slow accumulation of leaked leases. Each test asserts
the *cleanup* property, not just the happy path.
"""

from __future__ import annotations

import asyncio

import pytest

from phaseweaver import PhaseWeaver, ToolSpec
from phaseweaver.types import ContextState


async def _failing_tool_agent(unit) -> None:  # type: ignore[no-untyped-def]
    await unit.reason(prompt_tokens=256, out_tokens=16)
    result = await unit.call_tool(
        ToolSpec("boom", "python", timeout_s=5.0),
        {"code": "raise ValueError('boom')"},
    )
    assert result.status.value == "error"
    # A tool error is not fatal: the agent gets to reason about the failure.
    await unit.reason(prompt_tokens=300, out_tokens=16)


async def _slow_agent(unit) -> None:  # type: ignore[no-untyped-def]
    await unit.reason(prompt_tokens=512, out_tokens=32)
    await unit.call_tool(
        ToolSpec("slow", "python", expected_duration_s=60.0), {"code": "pass"}
    )
    await unit.reason(prompt_tokens=600, out_tokens=32)


def test_tool_error_is_not_fatal() -> None:
    async def scenario() -> PhaseWeaver:
        async with PhaseWeaver(
            profile="coding", capacity_gib=2.0, simulate_tools=False
        ) as pw:
            outcome = await pw.run("failing", _failing_tool_agent)
            assert outcome.phase == "COMPLETED"
            return pw

    pw = asyncio.run(scenario())
    assert not pw.runtime.ledger.leases


def test_cancelled_task_releases_everything() -> None:
    async def scenario() -> PhaseWeaver:
        async with PhaseWeaver(profile="coding", capacity_gib=2.0) as pw:
            task = asyncio.create_task(pw.run("victim", _slow_agent))
            clock = pw._clock
            await clock.wait(1.0)  # let it reach the tool phase
            task.cancel()
            outcome = await task
            assert outcome.phase == "CANCELLED"
            return pw

    pw = asyncio.run(scenario())
    assert not pw.runtime.broker.active_leases(), "cancel must not leak leases"
    for unit in pw.runtime.registry.all_units():
        if unit.context is not None:
            assert unit.context.state is ContextState.RELEASED


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_lease_leaks_across_seeds(seed: int) -> None:
    """Adversarial regression for the K15 family: strict invariants, three seeds."""

    async def scenario() -> PhaseWeaver:
        from phaseweaver.app import mock_caps
        from phaseweaver.backends.mock import MockBackend
        from phaseweaver.clock import VirtualClock
        from phaseweaver.leases import Budget
        from phaseweaver.policy import YieldOnToolPolicy
        from phaseweaver.runtime import Runtime, RuntimeConfig

        clock = VirtualClock()
        clock.start_driver()
        caps = mock_caps(2.0)
        rt = Runtime(
            backend=MockBackend(caps=caps, clock=clock),
            clock=clock,
            budget=Budget(total_bytes=caps.capacity_bytes, headroom_frac=0.05),
            policy=YieldOnToolPolicy(),
            config=RuntimeConfig(run_id=f"leak-{seed}", strict_invariants=True),
        )
        await rt.start()

        async def agent(unit) -> None:  # type: ignore[no-untyped-def]
            prompt = 4096 + seed * 128
            for _ in range(2):
                await unit.reason(prompt_tokens=prompt, out_tokens=64)
                await unit.call_tool(
                    ToolSpec("t", "python", expected_duration_s=3.0), {"code": "pass"}
                )
                prompt += 64

        tasks = []
        for i in range(4):
            tasks.append(asyncio.create_task(rt.run_task(f"t{i}", agent)))
            await clock.wait(0.3)
        await asyncio.gather(*tasks)
        await rt.shutdown()
        clock.stop_driver()
        return type("P", (), {"runtime": rt})()

    pw = asyncio.run(scenario())
    assert not pw.runtime.broker.active_leases()
    for unit in pw.runtime.registry.all_units():
        if unit.context is not None:
            assert unit.context.state is ContextState.RELEASED
