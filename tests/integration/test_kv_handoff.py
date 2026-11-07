"""K12: KV handoff - the child continues from the parent's blocks with no recompute."""

from __future__ import annotations

import asyncio

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.leases import Budget
from phaseweaver.policy import PinAlwaysPolicy
from phaseweaver.runtime import AgentHandle, Runtime, RuntimeConfig
from phaseweaver.types import ContextState


def caps() -> BackendCaps:
    return BackendCaps(
        capacity_bytes=8 * 1024 * 1024,
        block_tokens=16,
        bytes_per_token=1000,
        prefill_tok_per_s=1000,
        decode_tok_per_s=100,
        mem_bw_bps=10**7,
        supports_offload=True,
    )


async def child(handle: AgentHandle) -> None:
    await handle.reason(prompt_tokens=512, out_tokens=16)


async def parent(handle: AgentHandle) -> None:
    await handle.reason(prompt_tokens=256, out_tokens=16)
    await handle.fork("delegate", child, inherit="handoff")
    await handle.reason(prompt_tokens=320, out_tokens=16)


def run(inherit: bool) -> Runtime:
    async def scenario() -> Runtime:
        clock = VirtualClock()
        clock.start_driver()
        c = caps()
        rt = Runtime(
            backend=MockBackend(caps=c, clock=clock),
            clock=clock,
            budget=Budget(total_bytes=c.capacity_bytes, headroom_frac=0.0),
            policy=PinAlwaysPolicy(),
            config=RuntimeConfig(run_id="handoff"),
        )
        await rt.start()
        await rt.run_task("parent", _wrap(inherit), slo_s=600.0)
        await rt.shutdown()
        clock.stop_driver()
        return rt

    return asyncio.run(scenario())


def _wrap(inherit: bool):  # type: ignore[no-untyped-def]
    async def agent(handle: AgentHandle) -> None:
        await handle.reason(prompt_tokens=256, out_tokens=16)
        await handle.fork(
            "delegate", child, inherit="handoff" if inherit else "none"
        )
        await handle.reason(prompt_tokens=320, out_tokens=16)

    return agent


def test_handoff_avoids_recompute() -> None:
    rt = run(inherit=True)
    handed = [
        e
        for e in rt.journal.by_type("context_state_changed")
        if e.payload.get("state") == ContextState.HANDED_OFF.value
    ]
    assert handed, "parent must hand its KV to the child"
    recomputes = [
        int(e.payload.get("recompute_tokens", 0))
        for e in rt.journal.by_type("restore_completed")
    ]
    assert sum(recomputes) == 0, f"handoff must not recompute, got {recomputes}"


def test_without_handoff_parent_recomputes() -> None:
    rt = run(inherit=False)
    assert not [
        e
        for e in rt.journal.by_type("context_state_changed")
        if e.payload.get("state") == ContextState.HANDED_OFF.value
    ]
