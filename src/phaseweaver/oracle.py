"""B4: offline lower bound on makespan, consistent with the simulator's α model.

The mock shares one device across concurrent streams: a stream alone runs at
`base`, and with n streams each runs at `base / (1 + α(n-1))`. Aggregate
throughput is therefore `n * base / (1 + α(n-1))`, bounded by `base / α`.

A bound must be *optimistic* to be valid, so we use the most favourable
concurrency the capacity allows: `k = capacity / smallest_context`, capped by the
number of tasks. Two independent limits are combined:

    LB_device = max(total_prefill / A_prefill(k), total_decode / A_decode(k))
    LB_path   = max_i rounds_i * (prefill_i + decode_i + tool_i)
    LB        = max(LB_device, LB_path)

`LB_path` is the longest single-task critical path; `LB_device` is the best case
where the device is saturated the whole time. Both ignore admission and queueing,
so the reported gap is an **upper bound on the true optimality gap**.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaModel:
    prefill_tok_per_s: int = 3200
    decode_tok_per_s: int = 45
    alpha: float = 0.15
    capacity_bytes: int = 8 * 1024**3
    bytes_per_token: int = 57344

    def aggregate(self, n: int, base: float) -> float:
        n = max(1, n)
        return n * base / (1.0 + self.alpha * (n - 1))

    def max_streams(self, min_context_tokens: int, tasks: int) -> int:
        if min_context_tokens <= 0:
            return max(1, tasks)
        per_task = min_context_tokens * self.bytes_per_token
        if per_task <= 0:
            return max(1, tasks)
        return max(1, min(tasks, self.capacity_bytes // per_task))


def lower_bound_makespan(
    *,
    agents: int,
    rounds: int,
    prefill_tokens: int,
    decode_tokens: int,
    tool_s: float,
    model: AlphaModel,
) -> float:
    """Lower bound on wall-clock seconds for the whole workload."""
    if agents <= 0 or rounds <= 0:
        return 0.0
    total_prefill = agents * rounds * prefill_tokens
    total_decode = agents * rounds * decode_tokens

    k = model.max_streams(prefill_tokens, agents)
    agg_prefill = model.aggregate(k, model.prefill_tok_per_s)
    agg_decode = model.aggregate(k, model.decode_tok_per_s)

    device = max(total_prefill / agg_prefill, total_decode / agg_decode)
    path = rounds * (
        prefill_tokens / model.prefill_tok_per_s
        + decode_tokens / model.decode_tok_per_s
        + max(0.0, tool_s)
    )
    return max(device, path)
