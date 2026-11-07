from phaseweaver.oracle import AlphaModel, lower_bound_makespan


def model(**kw: object) -> AlphaModel:
    return AlphaModel(**kw)  # type: ignore[arg-type]


def test_single_task_is_critical_path() -> None:
    # 1s prefill + 1s decode + 10s tool, no contention with one task
    bound = lower_bound_makespan(
        agents=1,
        rounds=1,
        prefill_tokens=3200,
        decode_tokens=45,
        tool_s=10.0,
        model=model(),
    )
    assert bound == 12.0


def test_bound_is_below_serial_execution() -> None:
    # 10 identical tasks run serially would take 10 x 2s of device time.
    bound = lower_bound_makespan(
        agents=10,
        rounds=1,
        prefill_tokens=3200,
        decode_tokens=45,
        tool_s=0.0,
        model=model(capacity_bytes=10**12),
    )
    assert 0 < bound < 20.0


def test_tool_time_raises_the_bound() -> None:
    base = lower_bound_makespan(
        agents=8, rounds=2, prefill_tokens=4000, decode_tokens=100, tool_s=0.0, model=model()
    )
    with_tools = lower_bound_makespan(
        agents=8, rounds=2, prefill_tokens=4000, decode_tokens=100, tool_s=5.0, model=model()
    )
    assert with_tools > base


def test_small_capacity_raises_the_bound() -> None:
    small = model(capacity_bytes=4000 * 57344)
    large = model(capacity_bytes=10**12)
    kwargs = dict(agents=12, rounds=2, prefill_tokens=4000, decode_tokens=100, tool_s=2.0)
    assert lower_bound_makespan(model=small, **kwargs) >= lower_bound_makespan(
        model=large, **kwargs
    )  # type: ignore[arg-type]


def test_duty_cycle_dominates_when_tools_are_long() -> None:
    # With very long tools each task reasons a tiny fraction of its life, so the
    # concurrency ceiling - and therefore the bound - collapses.
    short_tools = lower_bound_makespan(
        agents=8, rounds=2, prefill_tokens=4000, decode_tokens=100, tool_s=1.0, model=model()
    )
    long_tools = lower_bound_makespan(
        agents=8, rounds=2, prefill_tokens=4000, decode_tokens=100, tool_s=60.0, model=model()
    )
    assert long_tools > short_tools
