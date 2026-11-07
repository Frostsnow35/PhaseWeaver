"""One agent turn: reasoning, activation, tool dispatch.

These three methods are what user agent code reaches through `AgentHandle`. They
live in their own mixin so `runtime.py` stays an orchestrator; the code is moved
verbatim (behaviour must not change), see docs/plans/2026-11-07-runtime-roadmap.md.
"""

from __future__ import annotations

import asyncio
from typing import Any

from phaseweaver.backends.base import InferenceBackend
from phaseweaver.clock import Clock
from phaseweaver.errors import AdmissionRejected
from phaseweaver.journal import EventJournal
from phaseweaver.leases import LeaseBroker
from phaseweaver.ledger import UnitLedger
from phaseweaver.policy import Scheduler
from phaseweaver.tools.base import ToolResult, ToolSpec, ToolStatus
from phaseweaver.tools.supervisor import ToolSupervisor
from phaseweaver.types import ContextState, InferenceContext, Phase
from phaseweaver.units import ExecutionUnit, UnitRegistry


class TurnsMixin:
    backend: InferenceBackend
    clock: Clock
    broker: LeaseBroker
    ledger: UnitLedger
    journal: EventJournal
    registry: UnitRegistry
    supervisor: ToolSupervisor
    policy: Scheduler
    config: Any

    def snapshot(self) -> Any:  # pragma: no cover - mixin contract
        raise NotImplementedError

    def _set_context(
        self,
        unit: ExecutionUnit,
        ctx: InferenceContext,
        state: ContextState,
        tokens: int | None = None,
    ) -> None:  # pragma: no cover - mixin contract
        raise NotImplementedError

    def _lease_bytes(self, tokens: int) -> int:  # pragma: no cover - mixin contract
        raise NotImplementedError

    def _max_tokens_for(self, unit: ExecutionUnit) -> int | None:  # pragma: no cover
        raise NotImplementedError

    async def _sync_lease(self, unit: ExecutionUnit, ctx: InferenceContext, want: int) -> None:
        raise NotImplementedError

    async def _park(self, unit: ExecutionUnit, ctx: InferenceContext, disposition: Any) -> None:
        raise NotImplementedError

    def _ensure_open(self) -> None:  # pragma: no cover - mixin contract
        raise NotImplementedError


    async def reason(self, unit: ExecutionUnit, prompt_tokens: int, out_tokens: int) -> Any:
        self._ensure_open()
        if unit.phase is Phase.QUEUED:
            self.registry.transition(unit.unit_id, Phase.REASONING)
        ctx = unit.context
        if ctx is None:
            ctx = InferenceContext(context_id=f"ctx-{unit.unit_id.replace('/', '_')}")
            unit.context = ctx
        if ctx.state is not ContextState.ACTIVE:
            await self._activate(unit, ctx, prompt_tokens, prompt_tokens + out_tokens)
        handle = self.ledger.handles[unit.unit_id]
        # A growing conversation: the engine keeps every token it has seen, so the
        # reservation must cover the union of caller-supplied prompt and history.
        tokens = max(prompt_tokens, ctx.resident_tokens) + out_tokens
        await self._sync_lease(unit, ctx, self._lease_bytes(tokens))
        # Count only units actually generating: units in REASONING may still be
        # waiting for capacity, so phase alone would overstate device concurrency.
        self.journal.append("engine_active", unit_id=unit.unit_id, payload={"delta": 1})
        try:
            stats = await asyncio.wait_for(
                self.backend.complete(handle, out_tokens),
                timeout=self.config.backend_timeout_s,
            )
        except TimeoutError as exc:
            # A stalled engine must fail this unit, not hang the whole tree.
            raise AdmissionRejected(
                f"backend timed out after {self.config.backend_timeout_s}s"
            ) from exc
        finally:
            self.journal.append("engine_active", unit_id=unit.unit_id, payload={"delta": -1})
        self._set_context(unit, ctx, ContextState.ACTIVE, tokens)
        return stats


    async def _activate(
        self, unit: ExecutionUnit, ctx: InferenceContext, prompt_tokens: int, reserve_tokens: int
    ) -> None:
        handle = self.ledger.handles.get(unit.unit_id)
        restored = None
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
                self.ledger.handles[unit.unit_id] = handle
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


    async def call_tool(
        self, unit: ExecutionUnit, spec: ToolSpec, args: dict[str, Any]
    ) -> ToolResult:
        self._ensure_open()
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
