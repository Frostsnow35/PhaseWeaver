from phaseweaver.oracle import AlphaModel, lower_bound_makespan


def model(**kw: object) -> AlphaModel:
    return AlphaModel(**kw)  # type: ignore[arg-type]


def test_single_task_equals_critical_path() -> None:
    bound = lower_bound_makespan(
        agents=1,
        rounds=1,
        prefill_tokens=3200,
        decode_tokens=45,
        tool_s=10.0,
        model=model(),
    )
    # 1s prefill + 1s decode + 10s tool
    assert bound == 12.0


def test_bound_scales_with_work_and_is_below_serial() -> None:
    m = model(capacity_bytes=10**12)  # plenty of streams
    bound = lower_bound_makespan(
        agents=10, rounds=1, prefill_tokens=3200, decode_tokens=45, tool_s=0.0, model=m
    )
    serial = 10 * (1.0 + 1.0)
    assert 0 < bound < serial


def test_capacity_limits_aggregate() -> None:
    small = model(capacity_bytes=3200 * 57344)  # room for exactly one context
    bound_small = lower_bound_makespan(
        agents=10, rounds=1, prefill_tokens=3200, decode_tokens=45, tool_s=0.0, model=small
    )
    big = model(capacity_bytes=10**12)
    bound_big = lower_bound_makespan(
        agents=10, rounds=1, prefill_tokens=3200, decode_tokens=45, tool_s=0.0, model=big
    )
    assert bound_small >= bound_big


def test_tool_time_enters_the_critical_path() -> None:
    base = lower_bound_makespan(
        agents=2, rounds=3, prefill_tokens=1000, decode_tokens=100, tool_s=0.0, model=model()
    )
    with_tools = lower_bound_makespan(
        agents=2, rounds=3, prefill_tokens=1000, decode_tokens=100, tool_s=5.0, model=model()
    )
    assert with_tools > base
