"""Execution unit tree: fork / join / tree-scoped cancel / phase transitions."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from phaseweaver.clock import Clock
from phaseweaver.errors import IllegalTransition, JoinBeforeTermination
from phaseweaver.journal import EventJournal
from phaseweaver.phases import check_phase_transition
from phaseweaver.types import (
    ForkMode,
    InferenceContext,
    Phase,
    UnitAggregate,
    UnitId,
    UnitKind,
)

CancelHook = Callable[[UnitId], Awaitable[None]]


@dataclass
class ExecutionUnit:
    unit_id: UnitId
    run_id: str
    kind: UnitKind
    label: str
    parent_id: UnitId | None = None
    mode: ForkMode = ForkMode.SHARE
    phase: Phase = Phase.PENDING
    context: InferenceContext | None = None
    priority: int = 5
    deadline: float | None = None
    arrival_ts: float = 0.0
    children: list[UnitId] = field(default_factory=list)
    aggregate: UnitAggregate = field(default_factory=UnitAggregate)

    @property
    def is_terminal(self) -> bool:
        return self.phase.is_terminal()


class UnitRegistry:
    def __init__(self, journal: EventJournal, clock: Clock) -> None:
        self._journal = journal
        self._clock = clock
        self._units: dict[UnitId, ExecutionUnit] = {}
        self._cancel_hooks: list[CancelHook] = []
        self._lock = asyncio.Lock()

    # --- creation ---------------------------------------------------------
    def create_root(
        self, label: str, *, priority: int = 5, deadline: float | None = None
    ) -> ExecutionUnit:
        unit_id = f"{self._journal.run_id}/root"
        unit = ExecutionUnit(
            unit_id=unit_id,
            run_id=self._journal.run_id,
            kind=UnitKind.AGENT,
            label=label,
            priority=priority,
            deadline=deadline,
            arrival_ts=self._clock.now(),
        )
        self._units[unit_id] = unit
        self._journal.append(
            "unit_created",
            unit_id=unit_id,
            payload={
                "kind": unit.kind.value,
                "label": label,
                "arrival_ts": unit.arrival_ts,
                "deadline": deadline,
                "priority": priority,
            },
        )
        return unit

    def fork(
        self,
        parent_id: UnitId,
        label: str,
        *,
        mode: ForkMode = ForkMode.SHARE,
        priority: int | None = None,
    ) -> ExecutionUnit:
        parent = self._units[parent_id]
        index = len(parent.children)
        unit_id = f"{parent_id}/sub-{index}"
        unit = ExecutionUnit(
            unit_id=unit_id,
            run_id=parent.run_id,
            kind=UnitKind.SUBAGENT,
            label=label,
            parent_id=parent_id,
            mode=mode,
            priority=priority if priority is not None else parent.priority,
            deadline=parent.deadline,
            arrival_ts=self._clock.now(),
        )
        self._units[unit_id] = unit
        parent.children.append(unit_id)
        self._journal.append(
            "unit_created",
            unit_id=unit_id,
            payload={
                "kind": unit.kind.value,
                "label": label,
                "parent_id": parent_id,
                "mode": mode.value,
                "arrival_ts": unit.arrival_ts,
                "deadline": unit.deadline,
            },
        )
        return unit

    def create_tool_call(self, parent_id: UnitId, label: str) -> ExecutionUnit:
        parent = self._units[parent_id]
        index = len(parent.children)
        unit_id = f"{parent_id}/tool-{index}"
        unit = ExecutionUnit(
            unit_id=unit_id,
            run_id=parent.run_id,
            kind=UnitKind.TOOL_CALL,
            label=label,
            parent_id=parent_id,
            priority=parent.priority,
            deadline=parent.deadline,
            arrival_ts=self._clock.now(),
        )
        self._units[unit_id] = unit
        parent.children.append(unit_id)
        self._journal.append(
            "unit_created",
            unit_id=unit_id,
            payload={"kind": unit.kind.value, "label": label, "parent_id": parent_id},
        )
        return unit

    # --- queries ----------------------------------------------------------
    def get(self, unit_id: UnitId) -> ExecutionUnit:
        return self._units[unit_id]

    def all_units(self) -> list[ExecutionUnit]:
        return list(self._units.values())

    def descendants(self, unit_id: UnitId) -> list[UnitId]:
        out: list[UnitId] = []
        stack = list(self._units[unit_id].children)
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(self._units[cur].children)
        return out

    def children_all_terminal(self, unit_id: UnitId) -> bool:
        return all(self._units[c].is_terminal for c in self.descendants(unit_id))

    # --- transitions ------------------------------------------------------
    def transition(
        self,
        unit_id: UnitId,
        dst: Phase,
        *,
        cause_seq: int | None = None,
        reason: str | None = None,
    ) -> None:
        unit = self._units[unit_id]
        src = unit.phase
        try:
            check_phase_transition(src, dst, kind=unit.kind)
        except IllegalTransition:
            self._journal.append(
                "transition_rejected",
                unit_id=unit_id,
                phase_from=src.value,
                phase_to=dst.value,
                cause_seq=cause_seq,
            )
            raise
        unit.phase = dst
        self._journal.append(
            "phase_transitioned",
            unit_id=unit_id,
            phase_from=src.value,
            phase_to=dst.value,
            cause_seq=cause_seq,
            payload={
                "from": src.value,
                "to": dst.value,
                "ts": self._clock.now(),
                "reason": reason,
                "kind": unit.kind.value,
            },
        )

    # --- join / cancel ----------------------------------------------------
    def join(self, child_id: UnitId) -> ExecutionUnit:
        child = self._units[child_id]
        if not child.is_terminal:
            raise JoinBeforeTermination(f"{child_id} is {child.phase}")
        parent = self._units[child.parent_id] if child.parent_id else None
        if parent is not None:
            parent.aggregate.recompute_tokens += child.aggregate.recompute_tokens
            parent.aggregate.park_count += child.aggregate.park_count
            parent.aggregate.tasks_completed += child.aggregate.tasks_completed
        self._journal.append(
            "lease_released",
            unit_id=child_id,
            payload={"reason": "join", "recompute_tokens": child.aggregate.recompute_tokens},
        )
        return child

    def register_cancel_hook(self, hook: CancelHook) -> None:
        self._cancel_hooks.append(hook)

    async def cancel(self, unit_id: UnitId, reason: str = "cancelled") -> None:
        """Depth-first: descendants first, then the unit itself; awaits every hook."""
        async with self._lock:
            targets = self.descendants(unit_id) + [unit_id]
            for uid in targets:
                unit = self._units[uid]
                if unit.is_terminal:
                    continue
                for hook in self._cancel_hooks:
                    await hook(uid)
                self.transition(uid, Phase.CANCELLED, reason=reason)
