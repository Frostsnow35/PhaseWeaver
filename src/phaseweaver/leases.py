"""Leases: disposition rights plus a cost model (ADR-0002).

The broker only decides and accounts; it never touches the engine.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from phaseweaver.clock import Clock
from phaseweaver.journal import EventJournal
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


class LeaseBroker:
    def __init__(self, journal: EventJournal, clock: Clock, budget: Budget) -> None:
        self._journal = journal
        self._clock = clock
        self._budget = budget
        self._leases: dict[LeaseId, Lease] = {}
        self._seq = itertools.count(1)

    @property
    def budget(self) -> Budget:
        return self._budget

    def used(self, kind: LeaseKind) -> int:
        return sum(l.amount for l in self._leases.values() if l.active and l.kind is kind)

    def available(self, kind: LeaseKind) -> int:
        if kind is LeaseKind.KV_BYTES:
            return self._budget.capacity() - self.used(kind)
        return 1_000_000

    def acquire(
        self,
        unit_id: str,
        context_id: str | None,
        kind: LeaseKind,
        amount: int,
        priority: int,
        expires_at: float | None = None,
    ) -> Lease | None:
        self._journal.append(
            "lease_requested",
            unit_id=unit_id,
            payload={"kind": kind.value, "amount": amount, "priority": priority},
        )
        if amount > self.available(kind):
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
        )
        self._leases[lease.lease_id] = lease
        self._journal.append(
            "lease_granted",
            unit_id=unit_id,
            payload={
                "lease_id": lease.lease_id,
                "kind": kind.value,
                "amount": amount,
                "context_id": context_id,
            },
        )
        return lease

    def release(self, lease_id: LeaseId, reason: str = "released") -> None:
        lease = self._leases.get(lease_id)
        if lease is None or not lease.active:
            return
        lease.active = False
        self._journal.append(
            "lease_released",
            unit_id=lease.unit_id,
            payload={"lease_id": lease_id, "kind": lease.kind.value, "reason": reason},
        )

    def active_leases(self, kind: LeaseKind | None = None) -> list[Lease]:
        return [l for l in self._leases.values() if l.active and (kind is None or l.kind is kind)]

    def lease_for_context(self, context_id: str, kind: LeaseKind) -> Lease | None:
        for lease in self._leases.values():
            if lease.active and lease.kind is kind and lease.context_id == context_id:
                return lease
        return None

    def release_all(self, reason: str = "shutdown") -> None:
        for lease in self.active_leases():
            self.release(lease.lease_id, reason=reason)
        self._journal.append("forced_release", payload={"reason": reason})
