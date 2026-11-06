"""Runtime: wires phases, leases, dispositions, tools and the journal together."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phaseweaver.backends.base import InferenceBackend
from phaseweaver.clock import Clock
from phaseweaver.errors import AdmissionRejected
from phaseweaver.invariant import InvariantChecker
from phaseweaver.journal import EventJournal
from phaseweaver.leases import Budget, Lease, LeaseBroker
from phaseweaver.policy import ClusterSnapshot, Scheduler
from phaseweaver.tools.base import ToolResult, ToolSpec, ToolStatus
from phaseweaver.tools.supervisor import ToolSupervisor
from phaseweaver.types import (
    ContextState,
    Disposition,
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

    async def fork(self, label: str, fn: AgentFn) -> TaskOutcome:
        return await self._rt.fork(self.unit, label, fn)


class Runtime:
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
        self.supervisor = ToolSupervisor(self.journal)
        self.checker = InvariantChecker(
            registry=self.registry,
            broker=self.broker,
            journal=self.journal,
            strict=self.config.strict_invariants,
            clock=clock,
        )
        self._handles: dict[UnitId, Any] = {}
        self._leases: dict[UnitId, Lease] = {}
        self._park_times: dict[UnitId, list[float]] = {}
        self._started = False

    # --- lifecycle ---------------------------------------------------------
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
            if not unit.phase.is_terminal():
                self.registry.transition(unit.unit_id, Phase.FAILED)
            err = f"{type(exc).__name__}: {exc}"
        finally:
            await self._release_unit(unit)
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
        return ClusterSnapshot(
            now=now,
            capacity_bytes=self.backend.caps().capacity_bytes,
            resident_bytes=self.broker.used(LeaseKind.KV_BYTES),
            queued=sum(1 for u in units if u.phase is Phase.QUEUED),
            reasoning=sum(1 for u in units if u.phase is Phase.REASONING),
            slack_by_unit=slack,
            recent_thrash={u: len(v) for u, v in self._park_times.items()},
        )

    async def reason(self, unit: ExecutionUnit, prompt_tokens: int, out_tokens: int) -> Any:
        if unit.phase is Phase.QUEUED:
            self.registry.transition(unit.unit_id, Phase.REASONING)
        ctx = unit.context
        if ctx is None:
            ctx = InferenceContext(context_id=f"ctx-{unit.unit_id.replace('/', '_')}")
            unit.context = ctx
        if ctx.state is not ContextState.ACTIVE:
            await self._activate(unit, ctx, prompt_tokens)
        handle = self._handles[unit.unit_id]
        stats = await self.backend.complete(handle, out_tokens)
        self._set_context(unit, ctx, ContextState.ACTIVE, prompt_tokens + out_tokens)
        self._sync_lease(unit, ctx)
        return stats

    async def _activate(self, unit: ExecutionUnit, ctx: InferenceContext, prompt_tokens: int) -> None:
        handle = self._handles.get(unit.unit_id)
        restored = None
        if handle is None:
            handle = await self.backend.submit(unit.unit_id, ctx.context_id, prompt_tokens)
            self._handles[unit.unit_id] = handle
        else:
            self.journal.append("restore_started", unit_id=unit.unit_id)
            restored = await self.backend.restore(handle)
        self._set_context(unit, ctx, ContextState.ACTIVE, handle.resident_tokens or prompt_tokens)
        self._sync_lease(unit, ctx)
        if restored is not None:
            unit.aggregate.recompute_tokens += restored.recompute_tokens
            self.journal.append(
                "restore_completed",
                unit_id=unit.unit_id,
                payload={
                    "recompute_tokens": restored.recompute_tokens,
                    "latency_ms": restored.latency_ms,
                },
            )
            recorder = getattr(self.policy, "record_restore", None)
            if recorder is not None:
                recorder(unit.unit_id, self.clock.now())

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
        elif state in (ContextState.DROPPED, ContextState.RELEASED):
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

    # --- tools -------------------------------------------------------------
    async def call_tool(
        self, unit: ExecutionUnit, spec: ToolSpec, args: dict[str, Any]
    ) -> ToolResult:
        if unit.phase is Phase.QUEUED:
            self.registry.transition(unit.unit_id, Phase.REASONING)
        ctx = unit.context
        expected = spec.expected_duration_s or self.config.default_expected_tool_s
        if ctx is not None and ctx.state is ContextState.ACTIVE:
            disposition = self.policy.disposition_for(unit.unit_id, self.snapshot(), expected)
            await self._park(unit, ctx, disposition)
        if unit.phase is Phase.REASONING:
            self.registry.transition(unit.unit_id, Phase.ACTING)
        tool_unit = self.registry.create_tool_call(unit.unit_id, spec.name)
        self.registry.transition(tool_unit.unit_id, Phase.ACTING)
        try:
            result = await self.supervisor.run(tool_unit.unit_id, spec, args)
        except asyncio.CancelledError:
            if not tool_unit.phase.is_terminal():
                self.registry.transition(tool_unit.unit_id, Phase.CANCELLED)
            raise
        if not tool_unit.phase.is_terminal():
            self.registry.transition(tool_unit.unit_id, Phase.COMPLETED)
        if unit.phase is Phase.ACTING:
            if result.status in (ToolStatus.OK, ToolStatus.ERROR):
                self.registry.transition(unit.unit_id, Phase.REASONING)
            else:  # timeout / killed / cancelled: the task fails, no faking success
                self.registry.transition(unit.unit_id, Phase.FAILED)
        return result

    # --- dispositions ------------------------------------------------------
    async def _park(
        self, unit: ExecutionUnit, ctx: InferenceContext, disposition: Disposition
    ) -> None:
        handle = self._handles.get(unit.unit_id)
        if handle is None:
            return
        self.journal.append(
            "park_started", unit_id=unit.unit_id, payload={"disposition": disposition.value}
        )
        result = await self.backend.park(handle, disposition)
        self._set_context(unit, ctx, self._state_for(result.applied))
        if result.applied is Disposition.RETAIN:
            pass  # I2: RETAINED must keep holding the KV lease
        else:
            lease = self._leases.pop(unit.unit_id, None)
            if lease is not None:
                self.broker.release(lease.lease_id, reason="park")
        if result.applied is Disposition.OFFLOAD and result.moved_bytes > 0:
            self.broker.acquire(
                unit.unit_id,
                ctx.context_id,
                LeaseKind.OFFLOAD_BYTES,
                result.moved_bytes,
                unit.priority,
            )
        self.journal.append(
            "park_completed",
            unit_id=unit.unit_id,
            payload={
                "applied": result.applied.value,
                "degraded_from": result.degraded_from.value if result.degraded_from else None,
                "moved_bytes": result.moved_bytes,
                "latency_ms": result.latency_ms,
            },
        )
        recorder = getattr(self.policy, "record_park", None)
        if recorder is not None:
            recorder(unit.unit_id, self.clock.now())
        self._detect_thrash(unit)

    @staticmethod
    def _state_for(disposition: Disposition) -> ContextState:
        return {
            Disposition.RETAIN: ContextState.RETAINED,
            Disposition.DROP: ContextState.DROPPED,
            Disposition.OFFLOAD: ContextState.OFFLOADED,
            Disposition.RELEASE: ContextState.RELEASED,
        }[disposition]

    def _detect_thrash(self, unit: ExecutionUnit) -> None:
        """Repeated parks inside the window raise a thrash event; the scheduler promotes."""
        policy_config = getattr(self.policy, "config", None)
        window_s = float(getattr(policy_config, "thrash_window_s", 60.0))
        now = self.clock.now()
        history = [ts for ts in self._park_times.get(unit.unit_id, []) if now - ts <= window_s]
        history.append(now)
        self._park_times[unit.unit_id] = history
        if len(history) >= 2:
            promoter = getattr(self.policy, "_state", None)
            if callable(promoter):
                promoter(unit.unit_id).park_count_window = len(history)
            self.journal.append(
                "thrash",
                unit_id=unit.unit_id,
                payload={"count": len(history), "window_s": window_s},
            )

    def _sync_lease(self, unit: ExecutionUnit, ctx: InferenceContext) -> None:
        want = ctx.resident_bytes
        current = self._leases.get(unit.unit_id)
        if current is not None and current.active and current.amount == want:
            return
        if current is not None:
            self.broker.release(current.lease_id, reason="resize")
            self._leases.pop(unit.unit_id, None)
        lease = self.broker.acquire(
            unit.unit_id, ctx.context_id, LeaseKind.KV_BYTES, want, unit.priority
        )
        if lease is None:
            raise AdmissionRejected(f"kv budget exhausted for {unit.unit_id}")
        self._leases[unit.unit_id] = lease

    async def _release_unit(self, unit: ExecutionUnit) -> None:
        self._leases.pop(unit.unit_id, None)
        for lease in self.broker.active_leases():
            if lease.unit_id == unit.unit_id:
                self.broker.release(lease.lease_id, reason="unit_end")
        if unit.context is not None and unit.context.state is not ContextState.RELEASED:
            self._set_context(unit, unit.context, ContextState.RELEASED)

    async def cancel(self, unit: ExecutionUnit, reason: str = "cancelled") -> None:
        if not unit.phase.is_terminal():
            await self.registry.cancel(unit.unit_id, reason=reason)
        await self._release_unit(unit)

    # --- subagents ---------------------------------------------------------
    async def fork(self, parent: ExecutionUnit, label: str, fn: AgentFn) -> TaskOutcome:
        child = self.registry.fork(parent.unit_id, label, mode=ForkMode.SHARE)
        started = self.clock.now()
        self.registry.transition(child.unit_id, Phase.QUEUED)
        await fn(AgentHandle(self, child))
        if not child.phase.is_terminal():
            self.registry.transition(child.unit_id, Phase.COMPLETED)
        await self._release_unit(child)
        self.registry.join(child.unit_id)
        return TaskOutcome(
            unit_id=child.unit_id,
            phase=child.phase.value,
            latency_ms=(self.clock.now() - started) * 1000,
        )
