"""K14: nested handoff and handoff under an ISOLATE quota."""

from __future__ import annotations

import asyncio

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.leases import Budget
from phaseweaver.policy import PinAlwaysPolicy
from phaseweaver.runtime import AgentHandle, Runtime, RuntimeConfig
from phaseweaver.types import ForkMode


def caps(capacity_mib: int = 8) -> BackendCaps:
    return BackendCaps(
        capacity_bytes=capacity_mib * 1024 * 1024,
        block_tokens=16,
        bytes_per_token=1000,
        prefill_tok_per_s=1000,
        decode_tok_per_s=100,
        mem_bw_bps=10**7,
        supports_offload=True,
    )


def total_recompute(rt: Runtime) -> int:
    return sum(
        int(e.payload.get("recompute_tokens", 0)) for e in rt.journal.by_type("restore_completed")
    )


def run(agent, *, capacity_mib: int = 8, quota: int = 512 * 1024) -> Runtime:  # type: ignore[no-untyped-def]
    async def scenario() -> Runtime:
        clock = VirtualClock()
        clock.start_driver()
        c = caps(capacity_mib)
        rt = Runtime(
            backend=MockBackend(caps=c, clock=clock),
            clock=clock,
            budget=Budget(total_bytes=c.capacity_bytes, headroom_frac=0.0),
            policy=PinAlwaysPolicy(),
            config=RuntimeConfig(run_id="nested", isolate_quota_bytes=quota),
        )
        await rt.start()
        await rt.run_task("root", agent, slo_s=600.0)
        await rt.shutdown()
        clock.stop_driver()
        return rt

    return asyncio.run(scenario())


async def _grandchild(handle: AgentHandle) -> None:
    await handle.reason(prompt_tokens=384, out_tokens=8)


async def _child(handle: AgentHandle) -> None:
    await handle.reason(prompt_tokens=320, out_tokens=8)
    await handle.fork("grandchild", _grandchild, inherit="handoff")


async def _root(handle: AgentHandle) -> None:
    await handle.reason(prompt_tokens=256, out_tokens=8)
    await handle.fork("child", _child, inherit="handoff")


def test_nested_handoff_chain() -> None:
    rt = run(_root)
    handed = [
        e
        for e in rt.journal.by_type("context_state_changed")
        if e.payload.get("state") == "HANDED_OFF"
    ]
    # root -> child and child -> grandchild
    assert len(handed) >= 2
    assert total_recompute(rt) == 0, "a handoff chain must not recompute"


def test_handoff_inside_isolate_quota() -> None:
    async def agent(handle: AgentHandle) -> None:
        await handle.reason(prompt_tokens=256, out_tokens=8)
        await handle.fork(
            "isolated-child",
            _grandchild,
            mode=ForkMode.ISOLATE,
            inherit="handoff",
        )

    rt = run(agent, quota=1024 * 1024)  # 1 MiB quota, child needs ~384 KiB
    scoped = [
        le
        for le in rt.broker.active_leases()
        if le.scope is not None
    ]
    assert not scoped, "child leases must be released after join"
    assert not rt.broker.active_leases()
