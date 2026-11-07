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

    def duty_streams(self, k: int, reasoning_s: float, tool_s: float) -> float:
        """Sustained concurrency under phase alternation.

        Each task spends `tool_s` away from the device every round, so with k
        resident contexts the average number of streams actually generating is at
        most `k * reasoning/(reasoning+tool)`. Staggering cannot beat this: the
        device can only serve contexts that are in their reasoning phase.
        """
        total = reasoning_s + max(0.0, tool_s)
        if total <= 0:
            return float(max(1, k))
        return max(0.0, min(float(k), k * reasoning_s / total))


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

    reasoning_s = (
        prefill_tokens / model.prefill_tok_per_s + decode_tokens / model.decode_tok_per_s
    )
    k = model.max_streams(prefill_tokens, agents)

    def device_bound(streams: int) -> float:
        return max(
            total_prefill / model.aggregate(streams, model.prefill_tok_per_s),
            total_decode / model.aggregate(streams, model.decode_tok_per_s),
        )

    # Valid lower bound: assume the device stays saturated at k streams.
    device = device_bound(k)
    path = rounds * (reasoning_s + max(0.0, tool_s))
    return max(device, path, 1e-9)


def duty_cycle_estimate(
    *,
    agents: int,
    rounds: int,
    prefill_tokens: int,
    decode_tokens: int,
    tool_s: float,
    model: AlphaModel,
) -> float:
    """Steady-state estimate, NOT a lower bound.

    If every task cycles reasoning -> tool in lockstep, the average number of
    streams that can be generating is `k * reasoning/(reasoning+tool)`. The real
    lower bound may be better than this (perfect staggering can fill the gaps),
    so treat this as a *prediction of the phase-aligned regime*, useful for
    explaining a gap - never as proof of optimality.
    """
    reasoning_s = (
        prefill_tokens / model.prefill_tok_per_s + decode_tokens / model.decode_tok_per_s
    )
    k = model.max_streams(prefill_tokens, agents)
    n_eff = model.duty_streams(k, reasoning_s, tool_s)
    if n_eff < 1.0:
        n_eff = 1.0
    total_prefill = agents * rounds * prefill_tokens
    total_decode = agents * rounds * decode_tokens
    return max(
        rules_device(total_prefill, total_decode, n_eff, model),
        rounds * (reasoning_s + max(0.0, tool_s)),
    )


def rules_device(
    total_prefill: int, total_decode: int, streams: float, model: AlphaModel
) -> float:
    """Estimated device seconds: prefill and decode work are added, because with a
    handful of streams the phases do not overlap enough to hide either one."""
    n = max(1, int(round(streams)))
    return total_prefill / model.aggregate(n, model.prefill_tok_per_s) + total_decode / (
        model.aggregate(n, model.decode_tok_per_s)
    )
