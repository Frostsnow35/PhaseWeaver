"""Invariant checks. strict mode raises; audit mode only records events."""

from __future__ import annotations

from dataclasses import dataclass

from phaseweaver.clock import Clock
from phaseweaver.errors import PhaseWeaverError
from phaseweaver.journal import EventJournal
from phaseweaver.leases import LeaseBroker
from phaseweaver.phases import NON_GPU_PHASES
from phaseweaver.types import ContextState, LeaseKind, Phase
from phaseweaver.units import UnitRegistry


@dataclass(frozen=True)
class Violation:
    code: str
    unit_id: str | None
    detail: str


class InvariantError(PhaseWeaverError):
    def __init__(self, violations: list[Violation]) -> None:
        super().__init__("; ".join(f"{v.code}:{v.unit_id}:{v.detail}" for v in violations))
        self.violations = violations


class InvariantChecker:
    def __init__(
        self,
        registry: UnitRegistry,
        broker: LeaseBroker,
        journal: EventJournal,
        *,
        strict: bool = True,
        clock: Clock | None = None,
    ) -> None:
        self._registry = registry
        self._broker = broker
        self._journal = journal
        self._strict = strict
        self._clock = clock

    def check(self) -> list[Violation]:
        violations: list[Violation] = []
        violations.extend(self._check_i1())
        violations.extend(self._check_i2())
        violations.extend(self._check_i3())
        violations.extend(self._check_i4())
        violations.extend(self._check_i8())
        for v in violations:
            self._journal.append(
                "invariant_violated",
                unit_id=v.unit_id,
                payload={"code": v.code, "detail": v.detail},
            )
        if violations and self._strict:
            raise InvariantError(violations)
        return violations

    # I1: no ACTIVE context in a non-GPU phase.
    def _check_i1(self) -> list[Violation]:
        out: list[Violation] = []
        for unit in self._registry.all_units():
            ctx = unit.context
            if ctx is None:
                continue
            if unit.phase in NON_GPU_PHASES and ctx.state is ContextState.ACTIVE:
                out.append(Violation("I1", unit.unit_id, f"{unit.phase} holds ACTIVE context"))
        return out

    # I2: GPU residency and KV lease are one-to-one.
    def _check_i2(self) -> list[Violation]:
        out: list[Violation] = []
        for unit in self._registry.all_units():
            ctx = unit.context
            if ctx is None:
                continue
            lease = self._broker.lease_for_context(ctx.context_id, LeaseKind.KV_BYTES)
            occupied = ctx.state in (ContextState.ACTIVE, ContextState.RETAINED)
            if occupied and lease is None:
                out.append(Violation("I2", unit.unit_id, "occupies GPU without lease"))
            if not occupied and lease is not None:
                out.append(Violation("I2", unit.unit_id, "lease without GPU residency"))
        return out

    # I3: never over-commit the budget.
    def _check_i3(self) -> list[Violation]:
        used = self._broker.used(LeaseKind.KV_BYTES)
        cap = self._broker.budget.capacity()
        if used > cap:
            return [Violation("I3", None, f"kv used {used} > capacity {cap}")]
        return []

    # I4: terminal units hold nothing.
    def _check_i4(self) -> list[Violation]:
        out: list[Violation] = []
        for unit in self._registry.all_units():
            if not unit.phase.is_terminal():
                continue
            held = [l for l in self._broker.active_leases() if l.unit_id == unit.unit_id]
            if held:
                out.append(Violation("I4", unit.unit_id, f"{len(held)} leases after terminal"))
            if unit.context is not None and unit.context.state is not ContextState.RELEASED:
                out.append(
                    Violation("I4", unit.unit_id, f"context {unit.context.state} after terminal")
                )
        return out

    # I8: an admitted unit must not wait forever (deadline + grace).
    def _check_i8(self) -> list[Violation]:
        if self._clock is None:
            return []
        out: list[Violation] = []
        now = self._clock.now()
        for unit in self._registry.all_units():
            if unit.phase is Phase.QUEUED and unit.deadline is not None:
                if now > unit.deadline + 60.0:
                    out.append(
                        Violation("I8", unit.unit_id, "queued beyond deadline + 60s")
                    )
        return out
