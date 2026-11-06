from phaseweaver.types import ContextState, InferenceContext, Phase


def test_terminal_phases() -> None:
    assert Phase.COMPLETED.is_terminal()
    assert Phase.FAILED.is_terminal()
    assert Phase.CANCELLED.is_terminal()
    assert not Phase.ACTING.is_terminal()


def test_context_defaults() -> None:
    ctx = InferenceContext(context_id="c1")
    assert ctx.state is ContextState.NONE
    assert ctx.resident_bytes == 0


def test_context_occupies_gpu() -> None:
    ctx = InferenceContext(context_id="c1", state=ContextState.ACTIVE, resident_bytes=4096)
    assert ctx.occupies_gpu
    ctx.state = ContextState.OFFLOADED
    assert not ctx.occupies_gpu
