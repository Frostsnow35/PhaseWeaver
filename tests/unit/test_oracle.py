from phaseweaver.oracle import OracleBudget, lower_bound_makespan


def test_bound_matches_single_task_serial_work() -> None:
    bound = lower_bound_makespan(
        tasks=[1], reasoning_s=[5.0], tool_s=[10.0], budget=OracleBudget(max_resident=2)
    )
    assert bound == 15.0


def test_bound_is_at_least_total_reasoning() -> None:
    bound = lower_bound_makespan(
        tasks=[1, 1, 1],
        reasoning_s=[5.0, 5.0, 5.0],
        tool_s=[10.0, 10.0, 10.0],
        budget=OracleBudget(max_resident=1),
    )
    assert bound == 15.0  # max(total reasoning 15, longest task 15)


def test_bound_never_exceeds_serial_total() -> None:
    bound = lower_bound_makespan(
        tasks=[2, 2], reasoning_s=[3.0, 4.0], tool_s=[1.0, 1.0], budget=OracleBudget(1)
    )
    assert bound == max(2 * 3.0 + 2 * 4.0, 2 * (4.0 + 1.0))
