import pytest

from phaseweaver.clock import VirtualClock
from phaseweaver.errors import IllegalTransition, JoinBeforeTermination
from phaseweaver.journal import EventJournal
from phaseweaver.types import ForkMode, Phase, UnitKind
from phaseweaver.units import UnitRegistry


def make_registry() -> tuple[UnitRegistry, EventJournal]:
    clock = VirtualClock()
    journal = EventJournal(run_id="r1", clock=clock)
    return UnitRegistry(journal=journal, clock=clock), journal


def test_create_root_and_fork() -> None:
    reg, _ = make_registry()
    root = reg.create_root("solve")
    child = reg.fork(root.unit_id, "sub-0", mode=ForkMode.SHARE)
    assert root.kind is UnitKind.AGENT
    assert child.kind is UnitKind.SUBAGENT
    assert child.unit_id.startswith(root.unit_id + "/")
    assert child.unit_id in reg.descendants(root.unit_id)


def test_transition_writes_exactly_one_event() -> None:
    reg, journal = make_registry()
    root = reg.create_root("solve")
    reg.transition(root.unit_id, Phase.QUEUED)
    transitions = journal.by_type("phase_transitioned")
    assert len(transitions) == 1
    assert transitions[0].payload["from"] == "PENDING"


def test_illegal_transition_rejected_and_logged() -> None:
    reg, journal = make_registry()
    root = reg.create_root("solve")
    reg.transition(root.unit_id, Phase.QUEUED)
    with pytest.raises(IllegalTransition):
        reg.transition(root.unit_id, Phase.ACTING)
    assert len(journal.by_type("transition_rejected")) == 1


def test_join_requires_terminal_child() -> None:
    reg, _ = make_registry()
    root = reg.create_root("solve")
    child = reg.fork(root.unit_id, "sub-0")
    with pytest.raises(JoinBeforeTermination):
        reg.join(child.unit_id)


def test_parent_terminal_implies_children_terminal() -> None:
    reg, _ = make_registry()
    root = reg.create_root("solve")
    child = reg.fork(root.unit_id, "sub-0")
    reg.transition(root.unit_id, Phase.QUEUED)
    reg.transition(root.unit_id, Phase.REASONING)
    assert not reg.children_all_terminal(root.unit_id)
    reg.transition(child.unit_id, Phase.QUEUED)
    reg.transition(child.unit_id, Phase.REASONING)
    reg.transition(child.unit_id, Phase.COMPLETED)
    assert reg.children_all_terminal(root.unit_id)


def test_tool_call_starts_acting_directly() -> None:
    reg, _ = make_registry()
    root = reg.create_root("solve")
    tool = reg.create_tool_call(root.unit_id, "shell")
    reg.transition(tool.unit_id, Phase.ACTING)
    assert tool.phase is Phase.ACTING
