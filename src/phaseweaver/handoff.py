"""KV handoff between units, in one place.

This is the most delicate part of the runtime, so it gets its own module and its
own rules:

1. **One await.** The engine call is the only suspension point. Everything that
   changes ownership (handle, lease, context state) happens synchronously after it,
   so no invariant checker can observe "residency without a lease" (K15).
2. **Two-way.** `handoff_to_child` transfers the parent's KV to the child;
   `handoff_from_child` transfers it back on join.
3. **All or nothing.** If the lease cannot follow the blocks (budget or ISOLATE
   quota), the blocks are handed back and the child re-prefills. Never leave blocks
   unleased.
"""

from __future__ import annotations

from typing import Any

from phaseweaver.backends.base import InferenceBackend
from phaseweaver.journal import EventJournal
from phaseweaver.leases import LeaseBroker
from phaseweaver.ledger import UnitLedger
from phaseweaver.types import ContextState, InferenceContext
from phaseweaver.units import ExecutionUnit


class HandoffMixin:
    #: provided by the composing Runtime
    backend: InferenceBackend
    broker: LeaseBroker
    ledger: UnitLedger
    journal: EventJournal

    def _lease_bytes(self, tokens: int) -> int:  # pragma: no cover - mixin contract
        raise NotImplementedError

    def _set_context(
        self,
        unit: ExecutionUnit,
        ctx: InferenceContext,
        state: ContextState,
        tokens: int | None = None,
    ) -> None:  # pragma: no cover - mixin contract
        raise NotImplementedError

    async def handoff_to_child(
        self, parent: ExecutionUnit, parent_ctx: Any, child: ExecutionUnit
    ) -> bool:
        """Give the parent's KV to the child. Returns True when it moved."""
        parent_id = parent.unit_id
        parent_handle = self.ledger.handles.get(parent_id)
        lease = self.ledger.leases.get(parent_id)
        if parent_handle is None or lease is None:
            return False
        child_ctx_id = f"ctx-{child.unit_id.replace('/', '_')}"
        tokens = parent_ctx.resident_tokens
        moved = await self.backend.handoff(parent_handle, child.unit_id, child_ctx_id)
        if not self.broker.transfer(
            lease.lease_id, child.unit_id, child_ctx_id, self._lease_bytes(tokens)
        ):
            # The lease cannot follow the blocks: give them back so they stay leased.
            await self.backend.handoff(moved, parent_id, parent_ctx.context_id)
            return False
        self.ledger.handles.pop(parent_id, None)
        self.ledger.handles[child.unit_id] = moved
        self.ledger.leases.pop(parent_id, None)
        self.ledger.leases[child.unit_id] = lease
        child.context = InferenceContext(context_id=child_ctx_id, resident_tokens=tokens)
        child.adopted_from = parent_id
        self._set_context(parent, parent_ctx, ContextState.HANDED_OFF)
        parent_ctx.handed_off_to = child.unit_id
        self.journal.append(
            "lease_transferred",
            unit_id=child.unit_id,
            payload={
                "from_unit": parent_id,
                "tokens": tokens,
                "direction": "to_child",
            },
        )
        return True

    async def handoff_from_child(
        self, parent: ExecutionUnit, parent_ctx: Any, child: ExecutionUnit
    ) -> bool:
        """Take the KV back on join. Returns True when it moved."""
        parent_id = parent.unit_id
        child_handle = self.ledger.handles.get(child.unit_id)
        child_lease = self.ledger.leases.get(child.unit_id)
        if (
            child.adopted_from is None
            or child_handle is None
            or child_lease is None
            or child.context is None
            or parent_ctx is None
            or parent_ctx.state is not ContextState.HANDED_OFF
        ):
            return False
        given_back = await self.backend.handoff(
            child_handle, parent_id, parent_ctx.context_id
        )
        tokens = child_handle.resident_tokens
        if not self.broker.transfer(
            child_lease.lease_id,
            parent_id,
            parent_ctx.context_id,
            self._lease_bytes(tokens),
        ):
            # Give the blocks back so they remain covered by a lease.
            await self.backend.handoff(given_back, child.unit_id, child.context.context_id)
            return False
        self.ledger.handles.pop(child.unit_id, None)
        self.ledger.leases.pop(child.unit_id, None)
        self.ledger.leases[parent_id] = child_lease
        self._set_context(child, child.context, ContextState.HANDED_OFF)
        self._set_context(parent, parent_ctx, ContextState.RETAINED)
        parent_ctx.handed_off_to = None
        self.journal.append(
            "lease_transferred",
            unit_id=parent_id,
            payload={
                "from_unit": child.unit_id,
                "tokens": tokens,
                "direction": "to_parent",
            },
        )
        return True
