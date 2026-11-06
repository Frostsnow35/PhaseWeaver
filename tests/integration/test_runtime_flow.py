import asyncio

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.leases import Budget
from phaseweaver.policy import PinAlwaysPolicy, YieldOnToolPolicy
from phaseweaver.runtime import AgentHandle, Runtime, RuntimeConfig
from phaseweaver.tools.base import ToolSpec


def caps() -> BackendCaps:
    return BackendCaps(
        capacity_bytes=4 * 1024 * 1024,
        block_tokens=16,
        bytes_per_token=1000,
        prefill_tok_per_s=1000,
        decode_tok_per_s=100,
        mem_bw_bps=10**7,
        supports_offload=True,
    )


async def agent(handle: AgentHandle) -> None:
    await handle.reason(prompt_tokens=256, out_tokens=32)
    await handle.call_tool(
        ToolSpec("sleepy", "python", expected_duration_s=30.0), {"code": "print(1)"}
    )
    await handle.reason(prompt_tokens=320, out_tokens=32)


async def run(policy: object) -> Runtime:
    clock = VirtualClock()
    clock.start_driver()
    backend = MockBackend(caps=caps(), clock=clock)
    rt = Runtime(
        backend=backend,
        clock=clock,
        budget=Budget(total_bytes=caps().capacity_bytes, headroom_frac=0.0),
        policy=policy,  # type: ignore[arg-type]
        config=RuntimeConfig(run_id="t"),
    )
    await rt.start()
    await rt.run_task("demo", agent, slo_s=600.0)
    await rt.shutdown()
    clock.stop_driver()
    return rt


def test_runtime_completes_and_releases() -> None:
    rt = asyncio.run(run(PinAlwaysPolicy()))
    assert rt.journal.by_type("phase_transitioned")
    assert rt.journal.by_type("context_state_changed")
    assert not rt.broker.active_leases(), "all leases must be released at unit end"


def test_yield_policy_parks_before_tool() -> None:
    rt = asyncio.run(run(YieldOnToolPolicy(retain_max_s=1.0, offload_max_s=60.0)))
    parked = rt.journal.by_type("park_completed")
    assert parked, "expected at least one park"
    assert rt.journal.by_type("restore_completed")
