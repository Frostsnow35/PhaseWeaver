"""Two transition tables: Phase (execution stage) and ContextState (inference context).

They are orthogonal (ADR-0001). This module is their single authoritative definition.
"""

from __future__ import annotations

from typing import Final

from phaseweaver.errors import IllegalTransition
from phaseweaver.types import ContextState, Phase, UnitKind

PHASE_TRANSITIONS: Final[dict[Phase, frozenset[Phase]]] = {
    # PENDING -> ACTING exists only for TOOL_CALL units (a tool never reasons);
    # UnitRegistry.transition passes the unit kind, and the guard below enforces it.
    Phase.PENDING: frozenset({Phase.QUEUED, Phase.ACTING, Phase.CANCELLED}),
    Phase.QUEUED: frozenset({Phase.REASONING, Phase.CANCELLED}),
    Phase.REASONING: frozenset(
        {
            Phase.ACTING,
            Phase.WAITING,
            Phase.COMPLETED,
            Phase.FAILED,
            Phase.CANCELLED,
        }
    ),
    Phase.ACTING: frozenset({Phase.REASONING, Phase.FAILED, Phase.CANCELLED}),
    Phase.WAITING: frozenset({Phase.REASONING, Phase.FAILED, Phase.CANCELLED}),
    Phase.COMPLETED: frozenset(),
    Phase.FAILED: frozenset(),
    Phase.CANCELLED: frozenset(),
}

CONTEXT_TRANSITIONS: Final[dict[ContextState, frozenset[ContextState]]] = {
    ContextState.NONE: frozenset({ContextState.ACTIVE, ContextState.RELEASED}),
    ContextState.ACTIVE: frozenset(
        {
            ContextState.ACTIVE,
            ContextState.RETAINED,
            ContextState.OFFLOADED,
            ContextState.DROPPED,
            ContextState.RELEASED,
        }
    ),
    ContextState.RETAINED: frozenset(
        {
            ContextState.ACTIVE,
            ContextState.OFFLOADED,
            ContextState.DROPPED,
            ContextState.RELEASED,
        }
    ),
    ContextState.OFFLOADED: frozenset({ContextState.ACTIVE, ContextState.RELEASED}),
    ContextState.DROPPED: frozenset({ContextState.ACTIVE, ContextState.RELEASED}),
    ContextState.RELEASED: frozenset(),
}

#: I1: these phases must never hold an ACTIVE context.
NON_GPU_PHASES: Final[frozenset[Phase]] = frozenset(
    {Phase.PENDING, Phase.QUEUED, Phase.ACTING, Phase.WAITING}
)


def check_phase_transition(src: Phase, dst: Phase, *, kind: UnitKind | None = None) -> None:
    if dst not in PHASE_TRANSITIONS[src]:
        raise IllegalTransition("phase", str(src), str(dst))
    if src is Phase.PENDING and dst is Phase.ACTING and kind is not UnitKind.TOOL_CALL:
        raise IllegalTransition("phase(kind=non-tool)", str(src), str(dst))


def check_context_transition(src: ContextState, dst: ContextState) -> None:
    if dst not in CONTEXT_TRANSITIONS[src]:
        raise IllegalTransition("context", str(src), str(dst))
