"""Leases: disposition rights plus a cost model (ADR-0002).

The broker only decides and accounts; it never touches the engine.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass

from phaseweaver.clock import Clock
from phaseweaver.journal import EventJournal
from phaseweaver.prefix import PrefixRegistry
from phaseweaver.types import LeaseId, LeaseKind


@dataclass(frozen=True)
class Budget:
    total_bytes: int
    headroom_frac: float = 0.10
    max_per_agent_factor: float = 1.5

    def capacity(self) -> int:
        return int(self.total_bytes * (1.0 - self.headroom_frac))


@dataclass
class Lease:
    lease_id: LeaseId
    unit_id: str
    context_id: str | None
    kind: LeaseKind
    amount: int
    priority: int
    expires_at: float | None = None
    active: bool = True
    #: ISOLATE quota scope (K8). None means "shared with the whole run".
    scope: str | None = None
    #: KV fork: prefix group this lease references (charged once per group).
    prefix_key: str | None = None
    #: bytes actually charged to the budget by this lease (0 for a shared prefix)
    charged: int = 0


class LeaseBroker:
    def __init__(self, journal: EventJournal, clock: Clock, budget: Budget) -> None:
        self._journal = journal
        self._clock = clock
        self._budget = budget
        self._leases: dict[LeaseId, Lease] = {}
        self._seq = itertools.count(1)
        #: ISOLATE quotas: scope id -> byte limit (K8).
        self._scopes: dict[str, int] = {}
        #: KV fork: shared prefixes are charged once, not per referencing unit.
        self.prefixes = PrefixRegistry()
        #: set whenever capacity is returned; waiters await this instead of
        #: polling the clock (polling defeats the virtual clock's idle jump).
        self._capacity_event = asyncio.Event()

    @property
    def budget(self) -> Budget:
        return self._budget

    def used(self, kind: LeaseKind) -> int:
        if kind is LeaseKind.KV_BYTES:
            # Prefix groups are charged once, and their charge lives as long as any
            # context references them - not as long as the lease that created them.
            return self.prefixes.unique_bytes() + sum(
                le.charged for le in self._leases.values() if le.active and le.kind is kind
            )
        return sum(le.charged for le in self._leases.values() if le.active and le.kind is kind)

    def available(self, kind: LeaseKind) -> int:
        if kind is LeaseKind.KV_BYTES:
            return self._budget.capacity() - self.used(kind)
        # Offload lives on CPU/disk: bounded by the same nominal budget, which
        # keeps the accounting honest without pretending it is GPU memory.
        return self._budget.total_bytes - self.used(kind)

    def open_scope(self, scope_id: str, limit_bytes: int) -> None:
        self._scopes[scope_id] = limit_bytes

    def close_scope(self, scope_id: str) -> None:
        self._scopes.pop(scope_id, None)

    def scope_limit(self, scope_id: str | None) -> int | None:
        if scope_id is None:
            return None
        return self._scopes.get(scope_id)

    def scope_usage(self, scope_id: str) -> int:
        return sum(
            le.amount
            for le in self._leases.values()
            if le.active and le.scope == scope_id and le.kind is LeaseKind.KV_BYTES
        )

    def acquire(
        self,
        unit_id: str,
        context_id: str | None,
        kind: LeaseKind,
        amount: int,
        priority: int,
        expires_at: float | None = None,
        scope: str | None = None,
        prefix_key: str | None = None,
        shared_bytes: int = 0,
    ) -> Lease | None:
        self._journal.append(
            "lease_requested",
            unit_id=unit_id,
            payload={
                "kind": kind.value,
                "amount": amount,
                "priority": priority,
                "scope": scope,
            },
        )
        limit = self.scope_limit(scope)
        over_quota = (
            limit is not None
            and scope is not None
            and amount > limit - self.scope_usage(scope)
        )
        if over_quota:
                self._journal.append(
                    "admission_rejected",
                    unit_id=unit_id,
                    payload={
                        "kind": kind.value,
                        "amount": amount,
                        "reason": "scope_quota",
                        "scope": scope,
                        "limit": limit,
                    },
                )
                return None
        charged = amount
        if prefix_key is not None:
            group_bytes = shared_bytes if shared_bytes > 0 else amount
            is_new_group = self.prefixes.get(prefix_key) is None
            self.prefixes.acquire(prefix_key, group_bytes, group_bytes, unit_id)
            # The group itself carries the charge; a lease only pays for its delta.
            charged = 0 if is_new_group else max(0, amount - shared_bytes)
        if charged > self.available(kind):
            self._journal.append(
                "admission_rejected",
                unit_id=unit_id,
                payload={"kind": kind.value, "amount": amount, "reason": "budget"},
            )
            return None
        lease = Lease(
            lease_id=f"L{next(self._seq)}",
            unit_id=unit_id,
            context_id=context_id,
            kind=kind,
            amount=amount,
            priority=priority,
            expires_at=expires_at,
            scope=scope,
            prefix_key=prefix_key,
            charged=charged,
        )
        self._leases[lease.lease_id] = lease
        self._journal.append(
            "lease_granted",
            unit_id=unit_id,
            payload={
                "lease_id": lease.lease_id,
                "kind": kind.value,
                "amount": amount,
                "charged": charged,
                "prefix": prefix_key,
                "context_id": context_id,
            },
        )
        return lease

    def resize(self, lease_id: LeaseId, new_amount: int) -> bool:
        """Atomically change a lease's amount.

        Never release-then-acquire: the gap would free budget while the engine
        still holds the blocks, so the engine could over-allocate (F1).
        """
        lease = self._leases.get(lease_id)
        if lease is None or not lease.active:
            return False
        delta = new_amount - lease.amount
        if lease.prefix_key is not None:
            # Shared-prefix leases grow with the conversation. Resizing must never
            # fail here: a failed resize would send the runtime into release+acquire
            # and open a window where residency and lease disagree (I2).
            if delta > 0:
                lease.charged += delta
            lease.amount = new_amount
            self._journal.append(
                "lease_granted",
                unit_id=lease.unit_id,
                payload={
                    "lease_id": lease.lease_id,
                    "kind": lease.kind.value,
                    "amount": new_amount,
                    "delta": delta,
                    "context_id": lease.context_id,
                    "resize_from": lease.amount,
                },
            )
            return True
        limit = self.scope_limit(lease.scope)
        over_scope = (
            limit is not None
            and lease.scope is not None
            and delta > limit - self.scope_usage(lease.scope)
        )
        if delta > 0 and (over_scope or delta > self.available(lease.kind)):
            return False
        old = lease.amount
        if delta > 0:
            # Growth is always charged to the lease; for a shared-prefix lease the
            # shared part is already charged once to its group.
            lease.charged += delta
        lease.amount = new_amount
        self._journal.append(
            "lease_granted",
            unit_id=lease.unit_id,
            payload={
                "lease_id": lease.lease_id,
                "kind": lease.kind.value,
                "amount": new_amount,
                "delta": delta,
                "context_id": lease.context_id,
                "resize_from": old,
            },
        )
        return True

    def release(self, lease_id: LeaseId, reason: str = "released") -> None:
        lease = self._leases.get(lease_id)
        if lease is None or not lease.active:
            return
        lease.active = False
        if lease.prefix_key is not None:
            self.prefixes.release(lease.prefix_key, lease.unit_id)
        self._capacity_event.set()
        self._journal.append(
            "lease_released",
            unit_id=lease.unit_id,
            payload={"lease_id": lease_id, "kind": lease.kind.value, "reason": reason},
        )

    def active_leases(self, kind: LeaseKind | None = None) -> list[Lease]:
        return [
            le
            for le in self._leases.values()
            if le.active and (kind is None or le.kind is kind)
        ]

    def lease_for_context(self, context_id: str, kind: LeaseKind) -> Lease | None:
        for lease in self._leases.values():
            if lease.active and lease.kind is kind and lease.context_id == context_id:
                return lease
        return None

    def release_all(self, reason: str = "shutdown") -> None:
        for lease in self.active_leases():
            self.release(lease.lease_id, reason=reason)
        self._journal.append("forced_release", payload={"reason": reason})

    async def wait_for_capacity(self, timeout_s: float = 120.0) -> None:
        """Block until some capacity is returned (event-driven, no polling)."""
        self._capacity_event.clear()
        try:
            await asyncio.wait_for(self._capacity_event.wait(), timeout=timeout_s)
        except TimeoutError:
            return
