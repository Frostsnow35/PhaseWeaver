"""I1 property test: random phase/state/amount combinations never violate I1 silently."""

from hypothesis import given, settings
from hypothesis import strategies as st

from phaseweaver.clock import VirtualClock
from phaseweaver.invariant import InvariantChecker
from phaseweaver.journal import EventJournal
from phaseweaver.leases import Budget, LeaseBroker
from phaseweaver.phases import NON_GPU_PHASES
from phaseweaver.types import ContextState, InferenceContext, LeaseKind, Phase
from phaseweaver.units import UnitRegistry


@settings(max_examples=50, deadline=None)
@given(
    phase=st.sampled_from(list(Phase)),
    state=st.sampled_from(list(ContextState)),
    amount=st.integers(min_value=0, max_value=1000),
)
def test_i1_no_active_context_in_non_gpu_phase(
    phase: Phase, state: ContextState, amount: int
) -> None:
    clock = VirtualClock()
    journal = EventJournal(run_id="r1", clock=clock)
    reg = UnitRegistry(journal=journal, clock=clock)
    broker = LeaseBroker(journal=journal, clock=clock, budget=Budget(2000, 0.0))
    unit = reg.create_root("x")
    unit.phase = phase
    unit.context = InferenceContext(context_id="c1", state=state, resident_bytes=amount)
    if state in (ContextState.ACTIVE, ContextState.RETAINED) and amount > 0:
        broker.acquire(unit.unit_id, "c1", LeaseKind.KV_BYTES, amount, 5)
    checker = InvariantChecker(registry=reg, broker=broker, journal=journal, strict=False)
    violations = checker.check()

    i1 = [v for v in violations if v.code == "I1"]
    if phase in NON_GPU_PHASES and state is ContextState.ACTIVE:
        assert i1, "I1 must fire for ACTIVE context in a non-GPU phase"
    else:
        assert not i1


@settings(max_examples=30, deadline=None)
@given(amounts=st.lists(st.integers(min_value=1, max_value=400), min_size=1, max_size=5))
def test_i3_budget_never_overcommitted(amounts: list[int]) -> None:
    clock = VirtualClock()
    journal = EventJournal(run_id="r1", clock=clock)
    reg = UnitRegistry(journal=journal, clock=clock)
    broker = LeaseBroker(journal=journal, clock=clock, budget=Budget(1000, 0.0))
    unit = reg.create_root("x")
    for amount in amounts:
        broker.acquire(unit.unit_id, "c1", LeaseKind.KV_BYTES, amount, 5)
    checker = InvariantChecker(registry=reg, broker=broker, journal=journal, strict=False)
    codes = [v.code for v in checker.check()]
    assert "I3" not in codes
    assert broker.used(LeaseKind.KV_BYTES) <= 1000
