import asyncio

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.leases import Budget
from phaseweaver.policy import YieldOnToolPolicy
from phaseweaver.runtime import AgentHandle, Runtime, RuntimeConfig
from phaseweaver.tools.base import ToolSpec
from phaseweaver.types import Phase


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
    await handle.reason(prompt_tokens=128, out_tokens=16)
    await handle.call_tool(ToolSpec("t", "python"), {"code": "pass"})


async def parent(handle: AgentHandle) -> None:
    await handle.reason(prompt_tokens=256, out_tokens=32)
    await asyncio.gather(*(handle.fork(f"sub-{i}", child) for i in range(2)))
    await handle.reason(prompt_tokens=320, out_tokens=32)


def test_fork_parks_parent_and_joins_children() -> None:
    async def scenario() -> Runtime:
        clock = VirtualClock()
        clock.start_driver()
        c = caps()
        rt = Runtime(
            backend=MockBackend(caps=c, clock=clock),
            clock=clock,
            budget=Budget(total_bytes=c.capacity_bytes, headroom_frac=0.0),
            policy=YieldOnToolPolicy(retain_max_s=1.0, offload_max_s=30.0),
            config=RuntimeConfig(run_id="fork"),
        )
        await rt.start()
        outcome = await rt.run_task("parent", parent, slo_s=600.0)
        await rt.shutdown()
        clock.stop_driver()
        assert outcome.phase == "COMPLETED"
        return rt

    rt = asyncio.run(scenario())

    waiting = [
        e
        for e in rt.journal.by_type("phase_transitioned")
        if e.payload.get("to") == "WAITING"
    ]
    assert waiting, "parent must enter WAITING while children run"

    created = [
        e
        for e in rt.journal.by_type("unit_created")
        if e.payload.get("kind") == "subagent"
    ]
    assert len(created) == 2

    joins = [e for e in rt.journal.by_type("lease_released") if e.payload.get("reason") == "join"]
    assert len(joins) == 2

    assert not rt.broker.active_leases(), "all leases released after join"
    for unit in rt.registry.all_units():
        if unit.kind.value == "subagent":
            assert unit.phase is Phase.COMPLETED
