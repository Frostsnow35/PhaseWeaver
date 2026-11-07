"""Teardown paths: releasing a unit and its subtree.

Split from `runtime.py` because cleanup has its own contract and its own failure
modes: it must be safe to call twice, safe under cancellation, and it must never
leave a lease or a context behind (I4), even when a child was still running.
"""

from __future__ import annotations

import asyncio
from typing import Any

from phaseweaver.backends.base import InferenceBackend
from phaseweaver.journal import EventJournal
from phaseweaver.leases import LeaseBroker
from phaseweaver.ledger import UnitLedger
from phaseweaver.types import ContextState, Phase
from phaseweaver.units import ExecutionUnit, UnitRegistry


class CleanupMixin:
    backend: InferenceBackend
    broker: LeaseBroker
    ledger: UnitLedger
    journal: EventJournal
    registry: UnitRegistry

    def _set_context(
        self,
        unit: ExecutionUnit,
        ctx: Any,
        state: ContextState,
        tokens: int | None = None,
    ) -> None:  # pragma: no cover - mixin contract
        raise NotImplementedError

    async def release_unit(self, unit: ExecutionUnit) -> None:
        """Release a unit and every descendant.

        Descendants are handled first and in reverse order so a child that is still
        mid-flight cannot keep a lease alive after its parent finishes (I5).
        """
        targets = [self.registry.get(uid) for uid in self.registry.descendants(unit.unit_id)]
        for target in reversed(targets + [unit]):
            await self._release_one(target)

    async def _release_one(self, unit: ExecutionUnit) -> None:
        unit.busy += 1
        try:
            if not unit.phase.is_terminal():
                self.registry.transition(unit.unit_id, Phase.CANCELLED, reason="parent_released")
            self.ledger.leases.pop(unit.unit_id, None)
            ctx_id = unit.context.context_id if unit.context is not None else None
            for lease in self.broker.active_leases():
                if lease.unit_id == unit.unit_id or (
                    ctx_id is not None and lease.context_id == ctx_id
                ):
                    self.broker.release(lease.lease_id, reason="unit_end")
            if unit.context is not None and unit.context.state is not ContextState.RELEASED:
                self._set_context(unit, unit.context, ContextState.RELEASED)
            handle = self.ledger.handles.pop(unit.unit_id, None)
            if handle is not None:
                # Bookkeeping above is already consistent; shield the engine call so
                # a cancellation can't leave the allocation half-released.
                await asyncio.shield(self.backend.abort(handle, "unit_end"))
        finally:
            unit.busy -= 1

    async def cancel(self, unit: ExecutionUnit, reason: str = "cancelled") -> None:
        if not unit.phase.is_terminal():
            await self.registry.cancel(unit.unit_id, reason=reason)
        await self.release_unit(unit)
