import pytest

from phaseweaver.errors import IllegalTransition
from phaseweaver.phases import (
    CONTEXT_TRANSITIONS,
    PHASE_TRANSITIONS,
    check_context_transition,
    check_phase_transition,
)
from phaseweaver.types import ContextState, Phase, UnitKind


def test_phase_legal_transition() -> None:
    check_phase_transition(Phase.ACTING, Phase.REASONING)


def test_phase_illegal_transition() -> None:
    with pytest.raises(IllegalTransition):
        check_phase_transition(Phase.COMPLETED, Phase.ACTING)


def test_all_phases_in_table() -> None:
    for phase in Phase:
        assert phase in PHASE_TRANSITIONS


def test_terminal_phases_have_no_exit() -> None:
    for phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        assert PHASE_TRANSITIONS[phase] == frozenset()


def test_pending_to_acting_is_tool_only() -> None:
    check_phase_transition(Phase.PENDING, Phase.ACTING, kind=UnitKind.TOOL_CALL)
    with pytest.raises(IllegalTransition):
        check_phase_transition(Phase.PENDING, Phase.ACTING, kind=UnitKind.AGENT)


def test_context_transition_dropped_can_restore() -> None:
    check_context_transition(ContextState.DROPPED, ContextState.ACTIVE)


def test_context_released_is_final() -> None:
    assert CONTEXT_TRANSITIONS[ContextState.RELEASED] == frozenset()
    with pytest.raises(IllegalTransition):
        check_context_transition(ContextState.RELEASED, ContextState.ACTIVE)
