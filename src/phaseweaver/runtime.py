"""Runtime: wires phases, leases, dispositions, tools and the journal together."""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phaseweaver.backends.base import InferenceBackend
from phaseweaver.cleanup import CleanupMixin
from phaseweaver.clock import Clock
from phaseweaver.errors import AdmissionRejected, PhaseWeaverError, RuntimeClosed
from phaseweaver.handoff import HandoffMixin
from phaseweaver.invariant import InvariantChecker
from phaseweaver.journal import EventJournal
from phaseweaver.leases import Budget, LeaseBroker
from phaseweaver.ledger import UnitLedger
from phaseweaver.parking import ParkingMixin
from phaseweaver.policy import ClusterSnapshot, Scheduler
from phaseweaver.tools.base import ToolResult, ToolSpec
from phaseweaver.tools.supervisor import ToolSupervisor
from phaseweaver.turns import TurnsMixin
from phaseweaver.types import (
    ContextState,
    ForkMode,
    InferenceContext,
    LeaseKind,
    Phase,
    UnitId,
)
from phaseweaver.units import ExecutionUnit, UnitRegistry

AgentFn = Callable[["AgentHandle"], Awaitable[None]]


@dataclass
class RuntimeConfig:
    run_id: str = "run"
    strict_invariants: bool = True
    journal_path: Path | None = None
    default_expected_tool_s: float = 5.0
    #: K8: hard cap for one ISOLATE child and its descendants.
    isolate_quota_bytes: int = 512 * 1024**2
    #: K11: when a quota cannot hold the whole context, compact instead of failing.
    compact_on_quota: bool = True
    #: Default KV inheritance for forks when the caller does not specify one.
    default_inherit_mode: str = "none"
    #: When true, unexpected exceptions (not PhaseWeaverError) propagate after
    #: cleanup instead of being reported as a failed task. Turn it on in tests and
    #: during development; leave it off in production so one agent can't kill the tree.
    strict_errors: bool = False
    #: How long a unit may wait for admission before the task fails.
    admission_wait_s: float = 60.0
    #: Hard ceiling for one backend call. A real engine can stall (driver reset, OOM
    #: retry loop); without this the whole agent tree waits forever.
    backend_timeout_s: float = 300.0
    #: Simulate tools against the virtual clock instead of spawning subprocesses.
    #: Deterministic and faithful to the declared tool latency; set False only when
    #: running against a real engine.
    simulate_tools: bool = True


@dataclass
class TaskOutcome:
    unit_id: UnitId
    phase: str
    latency_ms: float
    recompute_tokens: int = 0
    error: str | None = None


class AgentHandle:
    """The only surface user agent code touches."""

    def __init__(self, runtime: Runtime, unit: ExecutionUnit) -> None:
        self._rt = runtime
        self.unit = unit

    async def reason(self, *, prompt_tokens: int, out_tokens: int) -> Any:
        return await self._rt.reason(self.unit, prompt_tokens, out_tokens)

    async def call_tool(self, spec: ToolSpec, args: dict[str, Any]) -> ToolResult:
        return await self._rt.call_tool(self.unit, spec, args)

    async def fork(
        self,
        label: str,
        fn: AgentFn,
        *,
        mode: ForkMode = ForkMode.SHARE,
        inherit: str | None = None,
    ) -> TaskOutcome:
        return await self._rt.fork(self.unit, label, fn, mode=mode, inherit_mode=inherit)


class Runtime(ParkingMixin, HandoffMixin, CleanupMixin, TurnsMixin):
    def __init__(
        self,
        backend: InferenceBackend,
        clock: Clock,
        budget: Budget,
        policy: Scheduler,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.clock = clock
        self.backend = backend
        self.journal = EventJournal(
            run_id=self.config.run_id, clock=clock, path=self.config.journal_path
        )
        self.registry = UnitRegistry(journal=self.journal, clock=clock)
        self.broker = LeaseBroker(journal=self.journal, clock=clock, budget=budget)
        self.policy = policy
        self.supervisor = ToolSupervisor(
            self.journal, clock=clock, simulated=self.config.simulate_tools
        )
        self.checker = InvariantChecker(
            registry=self.registry,
            broker=self.broker,
            journal=self.journal,
            strict=self.config.strict_invariants,
            clock=clock,
        )
        self.ledger = UnitLedger()
        self._fork_depth: dict[UnitId, int] = {}
        self._started = False

    # --- lifecycle ---------------------------------------------------------
    def _ensure_open(self) -> None:
        if not self._started:
            raise RuntimeClosed(
                f"runtime {self.config.run_id!r} is not running; "
                "call await runtime.start() or use the PhaseWeaver facade"
            )

    async def start(self) -> None:
        if self._started:
            return
        self.registry.register_cancel_hook(self.supervisor.cancel_unit)
        self.journal.append("run_started", payload={"run_id": self.config.run_id})
        self._started = True

    async def shutdown(self) -> None:
        await self.supervisor.shutdown()
        self.broker.release_all()
        for unit in self.registry.all_units():
            if unit.context is not None and unit.context.state is not ContextState.RELEASED:
                unit.context.state = ContextState.RELEASED
        self.journal.append("run_finished")
        self.journal.close()
        self._started = False

    # --- task entry --------------------------------------------------------
    async def run_task(
        self, label: str, fn: AgentFn, *, slo_s: float = 120.0, priority: int = 5
    ) -> TaskOutcome:
        self._ensure_open()
        unit = self.registry.create_root(
            label, priority=priority, deadline=self.clock.now() + slo_s
        )
        started = self.clock.now()
        self.registry.transition(unit.unit_id, Phase.QUEUED)
        err: str | None = None
        try:
            await fn(AgentHandle(self, unit))
            if not unit.phase.is_terminal():
                self.registry.transition(unit.unit_id, Phase.COMPLETED)
        except asyncio.CancelledError:
            await self.cancel(unit, "task_cancelled")
            err = "cancelled"
        except Exception as exc:  # noqa: BLE001
            unexpected = not isinstance(exc, PhaseWeaverError)
            if unexpected:
                self.journal.append(
                    "run_error",
                    unit_id=unit.unit_id,
                    payload={
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-4000:],
                    },
                )
            if not unit.phase.is_terminal():
                self.registry.transition(unit.unit_id, Phase.FAILED, reason=type(exc).__name__)
            err = f"{type(exc).__name__}: {exc}"
            if unexpected and self.config.strict_errors:
                raise
        finally:
            await self.release_unit(unit)
        self.checker.check()
        return TaskOutcome(
            unit_id=unit.unit_id,
            phase=unit.phase.value,
            latency_ms=(self.clock.now() - started) * 1000,
            recompute_tokens=unit.aggregate.recompute_tokens,
            error=err,
        )

    # --- inference ---------------------------------------------------------
    def snapshot(self) -> ClusterSnapshot:
        units = self.registry.all_units()
        now = self.clock.now()
        slack = {
            u.unit_id: (u.deadline - now) if u.deadline is not None else 1e9 for u in units
        }
        caps = self.backend.caps()
        return ClusterSnapshot(
            now=now,
            capacity_bytes=caps.capacity_bytes,
            resident_bytes=self.broker.used(LeaseKind.KV_BYTES),
            queued=sum(1 for u in units if u.phase is Phase.QUEUED),
            reasoning=sum(1 for u in units if u.phase is Phase.REASONING),
            slack_by_unit=slack,
            recent_thrash={u: len(v) for u, v in self.ledger.park_times.items()},
            resident_bytes_by_unit={
                u.unit_id: (u.context.resident_bytes if u.context is not None else 0)
                for u in units
            },
            tokens_by_unit={
                u.unit_id: (u.context.resident_tokens if u.context is not None else 0)
                for u in units
            },
            prefill_tok_per_s=caps.prefill_tok_per_s,
            mem_bw_bps=caps.mem_bw_bps,
        )

    def _set_context(
        self,
        unit: ExecutionUnit,
        ctx: InferenceContext,
        state: ContextState,
        tokens: int | None = None,
    ) -> None:
        from phaseweaver.phases import check_context_transition

        check_context_transition(ctx.state, state)
        ctx.state = state
        if tokens is not None:
            ctx.resident_tokens = tokens
        if state in (ContextState.ACTIVE,):
            ctx.resident_bytes = ctx.resident_tokens * self.backend.caps().bytes_per_token
        elif state in (
            ContextState.DROPPED,
            ContextState.RELEASED,
            ContextState.OFFLOADED,
            ContextState.HANDED_OFF,
        ):
            # OFFLOADED still occupies CPU/disk, but nothing on the GPU.
            ctx.resident_bytes = 0
        self.journal.append(
            "context_state_changed",
            unit_id=unit.unit_id,
            payload={
                "context_id": ctx.context_id,
                "state": state.value,
                "resident_bytes": ctx.resident_bytes,
                "resident_tokens": ctx.resident_tokens,
            },
        )
        if state is ContextState.ACTIVE:
            notifier = getattr(self.policy, "note_activated", None)
            if notifier is not None:
                notifier(unit.unit_id, self.clock.now())

    # --- tools -------------------------------------------------------------
    async def fork(
        self,
        parent: ExecutionUnit,
        label: str,
        fn: AgentFn,
        *,
        mode: ForkMode = ForkMode.SHARE,
        inherit_mode: str | None = None,
    ) -> TaskOutcome:
        """Run a child to completion while the parent waits.

        The parent must not hold an ACTIVE context while in WAITING (I1), so it
        parks first; the next reason() restores lazily. A depth counter keeps the
        phase correct when several children are forked concurrently.

        ISOLATE (K8) gives the child tree its own hard quota so a runaway child
        cannot starve its siblings; SHARE lets it borrow from the global budget.
        """
        parent_id = parent.unit_id
        depth = self._fork_depth.get(parent_id, 0)
        scope_id: str | None = None
        if mode is ForkMode.ISOLATE:
            quota = self.config.isolate_quota_bytes
            # A quota is a *carve-out of total capacity*, not a claim on what is
            # free right now: otherwise forks fail precisely when the machine is
            # busy, which is when isolation is supposed to help.
            if quota > self.broker.budget.capacity():
                raise AdmissionRejected(
                    f"isolate quota {quota} exceeds total capacity for {label}"
                )
            scope_id = f"iso:{parent_id}:{depth}:{label}"
            self.broker.open_scope(scope_id, quota)
        # Reserve the slot *before* any await: concurrent fork() calls must not
        # both try to park the same parent context.
        self._fork_depth[parent_id] = depth + 1
        try:
            child = self.registry.fork(parent_id, label, mode=mode, scope=scope_id)
            inherit_mode = inherit_mode or self.config.default_inherit_mode
            relay = inherit_mode == "relay"
            can_inherit_mode = inherit_mode in ("handoff", "relay")
            parent_ctx = parent.context
            if self._inherits_now(parent_ctx, can_inherit_mode, depth, relay):
                await self.handoff_to_child(parent, parent_ctx, child)
        except BaseException:
            # Anything that raises before the main try/finally below (registry,
            # inheritance, backend handoff) must not leak the depth slot or the
            # ISOLATE scope, otherwise every later fork for this parent misbehaves.
            self._fork_depth[parent_id] = depth
            if scope_id is not None:
                self.broker.close_scope(scope_id)
            raise
        # KV inheritance. Two honest options:
        #   none    - the child re-prefills the parent's context (baseline)
        #   handoff - single-owner transfer: zero copy, zero recompute, but only
        #             the first child of a fork group can own it
        # (A "copy" mode was prototyped and dropped: duplicating the prefix doubles
        # residency and simply moves the failure to admission control.)
        # Relay (K12c): each child in turn takes the KV from its predecessor via the
        # parent, so every delegated task starts warm instead of only the first.
        parent_ctx = parent.context
        can_inherit_mode = inherit_mode in ("handoff", "relay")
        started = self.clock.now()
        try:
            await self._park_parent_for_fork(parent, depth)
            self.registry.transition(child.unit_id, Phase.QUEUED)
            error: str | None = None
            try:
                await fn(AgentHandle(self, child))
            except AdmissionRejected as exc:
                # A child that cannot fit its own quota is contained here: the
                # parent keeps running and the siblings keep their capacity.
                error = f"{type(exc).__name__}: {exc}"
            finally:
                # Always release and join, whatever path the child took: a child
                # left mid-flight would leak its lease (I4) forever.
                if not child.phase.is_terminal():
                    self.registry.transition(
                        child.unit_id, Phase.FAILED if error else Phase.COMPLETED
                    )
                if can_inherit_mode:
                    await self.handoff_from_child(parent, parent_ctx, child)
                await self.release_unit(child)
                self.registry.join(child.unit_id)
        finally:
            self._fork_depth[parent_id] -= 1
            if scope_id is not None:
                self.broker.close_scope(scope_id)
            if self._fork_depth[parent_id] == 0 and parent.phase is Phase.WAITING:
                self.registry.transition(parent_id, Phase.REASONING)
        return TaskOutcome(
            unit_id=child.unit_id,
            phase=child.phase.value,
            latency_ms=(self.clock.now() - started) * 1000,
        )

    @staticmethod
    def _inherits_now(
        parent_ctx: InferenceContext | None,
        can_inherit_mode: bool,
        depth: int,
        relay: bool,
    ) -> bool:
        """Only a warm context can be inherited, and only by the first child (or relay)."""
        if not can_inherit_mode or parent_ctx is None or parent_ctx.resident_tokens <= 0:
            return False
        if parent_ctx.state not in (ContextState.ACTIVE, ContextState.RETAINED):
            return False
        return depth == 0 or relay

    async def _park_parent_for_fork(self, parent: ExecutionUnit, depth: int) -> None:
        """I1: the parent must not hold an ACTIVE context while it is WAITING."""
        if depth != 0:
            return
        ctx = parent.context
        if ctx is not None and ctx.state is ContextState.ACTIVE:
            expected = self.config.default_expected_tool_s
            disposition = self.policy.disposition_for(parent.unit_id, self.snapshot(), expected)
            await self._park(parent, ctx, disposition)
        if parent.phase is Phase.REASONING:
            self.registry.transition(parent.unit_id, Phase.WAITING)