"""B4: offline lower bound on makespan. Experiment-only; never imported by runtime.

Bound used: for each task i with n_i rounds of (reasoning r_i, tool t_i),

    LB = max( sum_i n_i*r_i ,  max_i n_i*(r_i + t_i) )

The first term is the serial GPU work (reasoning never overlaps itself on one
device); the second is the longest single-task critical path. The bound ignores
capacity contention, so it is a *lower* bound and the reported optimality gap
is an upper bound on the true gap. Always report
`bound_kind=heuristic_lower_bound`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OracleBudget:
    max_resident: int
    max_nodes: int = 200_000


def lower_bound_makespan(
    tasks: list[int],
    reasoning_s: list[float],
    tool_s: list[float],
    budget: OracleBudget,
) -> float:
    if not tasks:
        return 0.0
    total_reasoning = sum(n * r for n, r in zip(tasks, reasoning_s, strict=True))
    longest_task = max(
        n * (r + t) for n, r, t in zip(tasks, reasoning_s, tool_s, strict=True)
    )
    return max(total_reasoning, longest_task)


# Backwards-compatible alias used by the plan document.
brute_force_min_makespan = lower_bound_makespan
