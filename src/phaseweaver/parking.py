"""Parking, dispositions and lease synchronisation.

Split out of `runtime.py` as a mixin: the methods only need the ledger, the broker
and the backend, and mixing them in keeps `self` semantics without touching any call
site. See docs/plans/2026-11-07-runtime-roadmap.md (R3).

Safety notes (learned the hard way, K15):
- park/restore must keep residency and lease in lockstep;
- `_sync_lease` treats the *engine* as the source of truth for capacity;
- `_state_for` is the only mapping from disposition to context state.
"""

from __future__ import annotations

from typing import Any

from phaseweaver.accounting import lease_bytes, max_tokens_for, record_park, state_for
from phaseweaver.backends.base import InferenceBackend
from phaseweaver.clock import Clock
from phaseweaver.errors import AdmissionRejected
from phaseweaver.journal import EventJournal
from phaseweaver.leases import LeaseBroker
from phaseweaver.ledger import UnitLedger
from phaseweaver.policy import ClusterSnapshot, Scheduler
from phaseweaver.types import (
    ContextState,
    Disposition,
    InferenceContext,
    LeaseKind,
)
from phaseweaver.units import ExecutionUnit


class ParkingMixin:
    #: provided by the composing Runtime
    backend: InferenceBackend
    broker: LeaseBroker
    ledger: UnitLedger
    journal: EventJournal
    clock: Clock
    policy: Scheduler
    config: Any

    def snapshot(self) -> ClusterSnapshot:  # pragma: no cover - overridden
        raise NotImplementedError

    def _set_context(
        self,
        unit: ExecutionUnit,
        ctx: InferenceContext,
        state: ContextState,
        tokens: int | None = None,
    ) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


    async def _park(
        self, unit: ExecutionUnit, ctx: InferenceContext, disposition: Disposition
    ) -> None:
        handle = self.ledger.handles.get(unit.unit_id)
        if handle is None:
            return
        unit.busy += 1
        try:
            self.journal.append(
                "park_started", unit_id=unit.unit_id, payload={"disposition": disposition.value}
            )
            result = await self.backend.park(handle, disposition)
            self._set_context(unit, ctx, state_for(result.applied))
            if result.applied is Disposition.RETAIN:
                pass  # I2: RETAINED must keep holding the KV lease
            else:
                lease = self.ledger.leases.pop(unit.unit_id, None)
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
        return state_for(disposition)


    def _detect_thrash(self, unit: ExecutionUnit) -> None:
        """Repeated parks inside the window raise a thrash event; the scheduler promotes."""
        policy_config = getattr(self.policy, "config", None)
        window_s = float(getattr(policy_config, "thrash_window_s", 60.0))
        now = self.clock.now()
        count = record_park(self.ledger.park_times, unit.unit_id, now, window_s)
        if count >= 2:
            promoter = getattr(self.policy, "_state", None)
            if callable(promoter):
                promoter(unit.unit_id).park_count_window = count
            self.journal.append(
                "thrash",
                unit_id=unit.unit_id,
                payload={"count": count, "window_s": window_s},
            )


    def _lease_bytes(self, tokens: int) -> int:
        return lease_bytes(self.backend.caps(), tokens)


    def _max_tokens_for(self, unit: ExecutionUnit) -> int | None:
        return max_tokens_for(self.backend.caps(), self.broker, unit.scope)


    async def _sync_lease(
        self, unit: ExecutionUnit, ctx: InferenceContext, want: int, *, wait_s: float = 600.0
    ) -> None:
        """Acquire (or atomically resize) the KV lease.

        Under pressure every resident unit wants to grow at the same time and
        nobody can, which deadlocks until the deadline (K1). The escape hatch:
        a unit that already holds a lease drops its own context (recompute later)
        so it can take the larger lease, instead of blocking everyone.
        """
        current = self.ledger.leases.get(unit.unit_id)
        if current is not None and current.active and current.amount == want:
            return
        limit = self.broker.scope_limit(unit.scope)
        if limit is not None and want > limit:
            handle = self.ledger.handles.get(unit.unit_id)
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
            # The engine is the ground truth: never promise blocks it cannot hold,
            # even if the byte-level budget still looks fine (K15).
            need = want if current is None else max(0, want - current.amount)
            if need > self.backend.available_bytes():
                if self.clock.now() >= deadline:
                    raise AdmissionRejected(f"engine capacity exhausted for {unit.unit_id}")
                await self.clock.wait(0.05)
                continue
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
                    self.ledger.leases[unit.unit_id] = lease
                    return
            else:
                # Atomic resize: the lease never disappears (F1).
                if self.broker.resize(current.lease_id, want):
                    return
                if want > current.amount and self.snapshot().pressure > 0.6:
                    # Self-yield: give the blocks back rather than deadlock.
                    self.broker.release(current.lease_id, reason="grow_self_drop")
                    self.ledger.leases.pop(unit.unit_id, None)
                    current = None
                    continue
            if self.clock.now() >= deadline:
                raise AdmissionRejected(f"kv budget exhausted for {unit.unit_id}")
            await self.clock.wait(0.05)

