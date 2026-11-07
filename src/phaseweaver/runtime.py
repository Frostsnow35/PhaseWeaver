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
    #: K8: hard cap for one ISOLATE child and its descendants.
    isolate_quota_bytes: int = 512 * 1024**2
    #: K11: when a quota cannot hold the whole context, compact instead of failing.
    compact_on_quota: bool = True
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
        inherit_prefix: bool = False,
    ) -> TaskOutcome:
        return await self._rt.fork(self.unit, label, fn, mode=mode, inherit_prefix=inherit_prefix)


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
        self._handles: dict[UnitId, Any] = {}
        self._leases: dict[UnitId, Lease] = {}
        self._park_times: dict[UnitId, list[float]] = {}
        self._fork_depth: dict[UnitId, int] = {}
        #: KV handoff handed back on join, picked up by the parent's next reason().
        self._pending_adopt: dict[UnitId, tuple[Any, int]] = {}
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
                self.registry.transition(unit.unit_id, Phase.FAILED, reason=type(exc).__name__)
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
        caps = self.backend.caps()
        return ClusterSnapshot(
            now=now,
            capacity_bytes=caps.capacity_bytes,
            resident_bytes=self.broker.used(LeaseKind.KV_BYTES),
            queued=sum(1 for u in units if u.phase is Phase.QUEUED),
            reasoning=sum(1 for u in units if u.phase is Phase.REASONING),
            slack_by_unit=slack,
            recent_thrash={u: len(v) for u, v in self._park_times.items()},
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

    async def reason(self, unit: ExecutionUnit, prompt_tokens: int, out_tokens: int) -> Any:
        if unit.phase is Phase.QUEUED:
            self.registry.transition(unit.unit_id, Phase.REASONING)
        ctx = unit.context
        if ctx is None:
            ctx = InferenceContext(context_id=f"ctx-{unit.unit_id.replace('/', '_')}")
            unit.context = ctx
        if ctx.state is not ContextState.ACTIVE:
            await self._activate(unit, ctx, prompt_tokens, prompt_tokens + out_tokens)
        handle = self._handles[unit.unit_id]
        # A growing conversation: the engine keeps every token it has seen, so the
        # reservation must cover the union of caller-supplied prompt and history.
        tokens = max(prompt_tokens, ctx.resident_tokens) + out_tokens
        await self._sync_lease(unit, ctx, self._lease_bytes(tokens))
        stats = await self.backend.complete(handle, out_tokens)
        self._set_context(unit, ctx, ContextState.ACTIVE, tokens)
        return stats

    async def _activate(
        self, unit: ExecutionUnit, ctx: InferenceContext, prompt_tokens: int, reserve_tokens: int
    ) -> None:
        handle = self._handles.get(unit.unit_id)
        restored = None
        pending = self._pending_adopt.pop(unit.unit_id, None)
        if pending is not None:
            adopted_handle, adopted_tokens = pending
            self._handles[unit.unit_id] = adopted_handle
            unit.busy += 1
            try:
                await self._sync_lease(unit, ctx, self._lease_bytes(adopted_tokens))
                self._set_context(unit, ctx, ContextState.ACTIVE, adopted_tokens)
            finally:
                unit.busy -= 1
            return
        cap_tokens = self._max_tokens_for(unit)
        if cap_tokens is not None and reserve_tokens > cap_tokens:
            # K11: do not fail an oversized child - compact its context instead.
            prompt_tokens = min(prompt_tokens, cap_tokens)
            reserve_tokens = min(reserve_tokens, cap_tokens)
            self.journal.append(
                "context_compacted",
                unit_id=unit.unit_id,
                payload={"to_tokens": reserve_tokens, "reason": "scope_quota"},
            )
        want = self._lease_bytes(reserve_tokens)
        unit.busy += 1
        try:
            # Lease first: the engine allocation must never outrun the budget.
            await self._sync_lease(unit, ctx, want)
            if handle is None:
                handle = await self.backend.submit(
                    unit.unit_id, ctx.context_id, prompt_tokens
                )
                self._handles[unit.unit_id] = handle
            else:
                self.journal.append("restore_started", unit_id=unit.unit_id)
                restored = await self.backend.restore(handle)
            self._set_context(
                unit, ctx, ContextState.ACTIVE, handle.resident_tokens or prompt_tokens
            )
        finally:
            unit.busy -= 1
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
        unit.busy += 1
        try:
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
        finally:
            unit.busy -= 1
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
        if recorder is not None and result.applied is not Disposition.RETAIN:
            recorder(unit.unit_id, self.clock.now())
        if result.applied is not Disposition.RETAIN:
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

    def _lease_bytes(self, tokens: int) -> int:
        """Round up to engine block granularity so the byte budget and the
        engine's block allocator can never disagree."""
        caps = self.backend.caps()
        blocks = (tokens + caps.block_tokens - 1) // caps.block_tokens
        return blocks * caps.block_tokens * caps.bytes_per_token

    def _max_tokens_for(self, unit: ExecutionUnit) -> int | None:
        """Largest context the unit's own quota can hold, in tokens."""
        limit = self.broker.scope_limit(unit.scope)
        if limit is None:
            return None
        caps = self.backend.caps()
        blocks = limit // (caps.block_tokens * caps.bytes_per_token)
        return max(caps.block_tokens, blocks * caps.block_tokens)

    async def _sync_lease(
        self, unit: ExecutionUnit, ctx: InferenceContext, want: int, *, wait_s: float = 600.0
    ) -> None:
        """Acquire (or atomically resize) the KV lease.

        Under pressure every resident unit wants to grow at the same time and
        nobody can, which deadlocks until the deadline (K1). The escape hatch:
        a unit that already holds a lease drops its own context (recompute later)
        so it can take the larger lease, instead of blocking everyone.
        """
        current = self._leases.get(unit.unit_id)
        if current is not None and current.active and current.amount == want:
            return
        limit = self.broker.scope_limit(unit.scope)
        if limit is not None and want > limit:
            handle = self._handles.get(unit.unit_id)
            cap_tokens = self._max_tokens_for(unit)
            if not self.config.compact_on_quota or handle is None or not cap_tokens:
                # A per-child cap cannot be satisfied by waiting for global capacity.
                raise AdmissionRejected(
                    f"isolate quota {limit} < required {want} for {unit.unit_id}"
                )
            new_tokens = await self.backend.truncate(handle, cap_tokens)
            ctx.resident_tokens = new_tokens
            want = self._lease_bytes(new_tokens)
            self.journal.append(
                "context_compacted",
                unit_id=unit.unit_id,
                payload={"to_tokens": new_tokens, "reason": "scope_quota_grow"},
            )
        deadline = self.clock.now() + wait_s
        while True:
            if current is None:
                lease = self.broker.acquire(
                    unit.unit_id,
                    ctx.context_id,
                    LeaseKind.KV_BYTES,
                    want,
                    unit.priority,
                    scope=unit.scope,
                )
                if lease is not None:
                    self._leases[unit.unit_id] = lease
                    return
            else:
                # Atomic resize: the lease never disappears (F1).
                if self.broker.resize(current.lease_id, want):
                    return
                if want > current.amount and self.snapshot().pressure > 0.6:
                    # Self-yield: give the blocks back rather than deadlock.
                    self.broker.release(current.lease_id, reason="grow_self_drop")
                    self._leases.pop(unit.unit_id, None)
                    current = None
                    continue
            if self.clock.now() >= deadline:
                raise AdmissionRejected(f"kv budget exhausted for {unit.unit_id}")
            await self.clock.wait(0.05)

    async def _release_unit(self, unit: ExecutionUnit) -> None:
        # Bookkeeping first, engine call last: the invariant checker must never
        # observe a terminal unit that still holds a lease (I4) or a lease without
        # residency (I2) while we await the backend.
        #
        # Recurse into descendants: a child left mid-flight by a failed parent
        # would otherwise leak its lease forever (I5).
        targets = [self.registry.get(uid) for uid in self.registry.descendants(unit.unit_id)]
        for target in reversed(targets + [unit]):
            await self._release_one(target)

    async def _release_one(self, unit: ExecutionUnit) -> None:
        unit.busy += 1
        try:
            if not unit.phase.is_terminal():
                self.registry.transition(unit.unit_id, Phase.CANCELLED, reason="parent_released")
            self._leases.pop(unit.unit_id, None)
            for lease in self.broker.active_leases():
                if lease.unit_id == unit.unit_id:
                    self.broker.release(lease.lease_id, reason="unit_end")
            if unit.context is not None and unit.context.state is not ContextState.RELEASED:
                self._set_context(unit, unit.context, ContextState.RELEASED)
            handle = self._handles.pop(unit.unit_id, None)
            if handle is not None:
                await self.backend.abort(handle, "unit_end")
        finally:
            unit.busy -= 1

    async def cancel(self, unit: ExecutionUnit, reason: str = "cancelled") -> None:
        if not unit.phase.is_terminal():
            await self.registry.cancel(unit.unit_id, reason=reason)
        await self._release_unit(unit)

    # --- subagents ---------------------------------------------------------
    async def fork(
        self,
        parent: ExecutionUnit,
        label: str,
        fn: AgentFn,
        *,
        mode: ForkMode = ForkMode.SHARE,
        inherit_prefix: bool = False,
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
        child = self.registry.fork(parent_id, label, mode=mode, scope=scope_id)
        # KV handoff: the parent transfers ownership of its resident blocks to the
        # child instead of copying them (or dropping and recomputing). Only the
        # first child of a fork group can inherit - one context, one owner.
        parent_ctx = parent.context
        if (
            inherit_prefix
            and depth == 0
            and parent_ctx is not None
            and parent_ctx.state in (ContextState.ACTIVE, ContextState.RETAINED)
        ):
            parent_handle = self._handles.get(parent_id)
            if parent_handle is not None:
                child_ctx_id = f"ctx-{child.unit_id.replace('/', '_')}"
                moved = await self.backend.handoff(
                    parent_handle, child.unit_id, child_ctx_id
                )
                self._handles.pop(parent_id, None)
                self._handles[child.unit_id] = moved
                child.context = InferenceContext(
                    context_id=child_ctx_id,
                    resident_tokens=parent_ctx.resident_tokens,
                )
                child.adopted_from = parent_id
                lease = self._leases.pop(parent_id, None)
                if lease is not None:
                    self.broker.release(lease.lease_id, reason="handoff")
                self._set_context(parent, parent_ctx, ContextState.HANDED_OFF)
                parent_ctx.handed_off_to = child.unit_id
                self.journal.append(
                    "context_state_changed",
                    unit_id=parent_id,
                    payload={
                        "context_id": parent_ctx.context_id,
                        "state": ContextState.HANDED_OFF.value,
                        "handed_off_to": child.unit_id,
                        "resident_bytes": 0,
                        "resident_tokens": parent_ctx.resident_tokens,
                    },
                )
        started = self.clock.now()
        try:
            if depth == 0:
                ctx = parent.context
                if ctx is not None and ctx.state is ContextState.ACTIVE:
                    expected = self.config.default_expected_tool_s
                    disposition = self.policy.disposition_for(
                        parent_id, self.snapshot(), expected
                    )
                    await self._park(parent, ctx, disposition)
                if parent.phase is Phase.REASONING:
                    self.registry.transition(parent_id, Phase.WAITING)
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
                # Hand the KV back before releasing: on join the parent resumes from
                # the child's context with zero recompute.
                child_handle = self._handles.get(child.unit_id)
                if (
                    child.adopted_from is not None
                    and child_handle is not None
                    and parent_ctx is not None
                    and parent_ctx.state is ContextState.HANDED_OFF
                ):
                    given_back = await self.backend.handoff(
                        child_handle, parent_id, parent_ctx.context_id
                    )
                    self._handles.pop(child.unit_id, None)
                    self._pending_adopt[parent_id] = (
                        given_back,
                        child_handle.resident_tokens,
                    )
                await self._release_unit(child)
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
