"""L1 entry point: one object that wires a profile into a running runtime.

    async with PhaseWeaver(profile="coding", capacity_gib=2.0) as pw:
        outcome = await pw.run("summarize-repo", agent_fn)
        print(pw.report().summary_row())

The facade is deliberately thin: it assembles `Runtime` from a `Profile` and
leaves every knob reachable. If you need control, drop to `Runtime` directly -
the facade has no behaviour of its own beyond defaults.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phaseweaver.autopilot import AutopilotPolicy
from phaseweaver.backends.base import BackendCaps, InferenceBackend
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import Clock, RealClock, VirtualClock
from phaseweaver.leases import Budget
from phaseweaver.metrics import MetricsReducer, RunReport
from phaseweaver.policy import PinAlwaysPolicy, Scheduler, TtlEvictPolicy, YieldOnToolPolicy
from phaseweaver.profile import Profile, resolve
from phaseweaver.runtime import AgentFn, Runtime, RuntimeConfig, TaskOutcome
from phaseweaver.scheduler import SchedulerConfig, SystemScheduler

DEFAULT_CAPACITY_GIB = 8.0
BYTES_PER_TOKEN = 57344  # Qwen2.5-7B bf16 KV per token
BLOCK_TOKENS = 16


def mock_caps(capacity_gib: float = DEFAULT_CAPACITY_GIB) -> BackendCaps:
    return BackendCaps(
        capacity_bytes=int(capacity_gib * 1024**3),
        block_tokens=BLOCK_TOKENS,
        bytes_per_token=BYTES_PER_TOKEN,
        prefill_tok_per_s=3200,
        decode_tok_per_s=45,
        mem_bw_bps=int(12 * 1024**3),
        supports_offload=True,
    )


def build_policy(name: str) -> Scheduler:
    if name == "auto":
        return AutopilotPolicy()
    if name == "B0":
        return PinAlwaysPolicy()
    if name == "B1":
        return TtlEvictPolicy()
    if name == "B2":
        return SystemScheduler(SchedulerConfig())
    if name == "B2a":
        return YieldOnToolPolicy()
    raise ValueError(f"unknown policy {name!r}")


@dataclass
class PhaseWeaver:
    """Context-managed runtime facade for a scenario profile."""

    profile: str | Profile = "coding"
    capacity_gib: float = DEFAULT_CAPACITY_GIB
    backend: InferenceBackend | None = None
    clock: Clock | None = None
    journal_path: Path | None = None
    run_id: str = "phaseweaver"
    simulate_tools: bool | None = None

    def __post_init__(self) -> None:
        self._profile: Profile = resolve(self.profile)
        self._owns_clock = self.clock is None
        if self.backend is None:
            self._clock: Clock = VirtualClock()
            self._backend: InferenceBackend = MockBackend(
                caps=mock_caps(self.capacity_gib), clock=self._clock
            )
            # A simulated backend only makes sense with simulated tools + virtual
            # time; a real backend implies real time and real tool execution.
            simulate = True if self.simulate_tools is None else self.simulate_tools
        else:
            self._clock = RealClock()
            self._backend = self.backend
            simulate = False if self.simulate_tools is None else self.simulate_tools
        if self.clock is not None:
            self._clock = self.clock
        self._simulate_tools = simulate
        self._runtime: Runtime | None = None

    # --- lifecycle ---------------------------------------------------------
    @property
    def runtime(self) -> Runtime:
        if self._runtime is None:
            raise RuntimeError("PhaseWeaver must be entered before use")
        return self._runtime

    async def __aenter__(self) -> PhaseWeaver:
        if isinstance(self._clock, VirtualClock):
            self._clock.start_driver()
        caps = self._backend.caps()
        self._runtime = Runtime(
            backend=self._backend,
            clock=self._clock,
            budget=Budget(
                total_bytes=caps.capacity_bytes, headroom_frac=self._profile.headroom_frac
            ),
            policy=build_policy(self._profile.policy),
            config=RuntimeConfig(
                run_id=self.run_id,
                journal_path=self.journal_path,
                simulate_tools=self._simulate_tools,
                default_inherit_mode=self._profile.inherit_mode,
            ),
        )
        await self._runtime.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._runtime is not None:
            await self._runtime.shutdown()
        if isinstance(self._clock, VirtualClock):
            self._clock.stop_driver()

    # --- the one call most users need --------------------------------------
    async def run(
        self,
        label: str,
        fn: AgentFn,
        *,
        slo_s: float | None = None,
        priority: int = 5,
    ) -> TaskOutcome:
        return await self.runtime.run_task(
            label,
            fn,
            slo_s=self._profile.slo_s if slo_s is None else slo_s,
            priority=priority,
        )

    async def run_many(self, tasks: list[tuple[str, AgentFn]]) -> list[TaskOutcome]:
        """Run independent tasks concurrently - the common agent-fleet shape."""
        coros = [self.run(label, fn) for label, fn in tasks]
        return list(await asyncio.gather(*coros))

    def report(self) -> RunReport:
        return MetricsReducer.replay(self.runtime.journal)

    def summary(self) -> dict[str, Any]:
        return self.report().summary_row()


async def run(
    label: str,
    fn: AgentFn,
    *,
    profile: str | Profile = "coding",
    capacity_gib: float = DEFAULT_CAPACITY_GIB,
) -> TaskOutcome:
    """One-shot convenience: run a single task with a profile and shut down."""
    async with PhaseWeaver(profile=profile, capacity_gib=capacity_gib) as pw:
        return await pw.run(label, fn)
