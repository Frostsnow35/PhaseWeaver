"""B4: a *provable* lower bound on makespan for the simulator's α model.

Derivation
----------
Let each task run `rounds` cycles of (reasoning r seconds, tool t seconds), so a
task reasons for at most the fraction `r / (r + t)` of the wall clock. With n
tasks sharing the same wall clock T:

    ∫ c(t) dt  =  total reasoning time  ≤  n · T · r/(r+t)

where `c(t)` is the number of streams generating at time t. Hence the average
concurrency is bounded by `c̄ ≤ n · r/(r+t)`, and because capacity allows at most
`k = floor(capacity / context_bytes)` resident contexts, `c̄ ≤ k · r/(r+t)`.

The device's token throughput at concurrency c is `A(c) = c·base/(1+α(c-1))`,
which is increasing. Work conservation gives `W = ∫ A(c) dt`; since A is
increasing, `∫ A(c) dt ≤ T · A(c̄)`, therefore

    T  ≥  W / A(c̄_max),        c̄_max = k · r/(r+t)

with W split into prefill and decode work (different base rates, added because a
stream cannot prefill and decode at the same time). The single-task critical path
`rounds·(r+t)` is a second, independent bound; the answer is the max of the two.

This is valid when tasks are homogeneous (same r, t); for heterogeneous workloads
we use the per-task minimum `r/(r+t)` and the smallest `r`, which keeps it a bound.
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

    def aggregate(self, n: float, base: float) -> float:
        n = max(1.0, n)
        return n * base / (1.0 + self.alpha * (n - 1))

    def max_streams(self, context_tokens: int, tasks: int) -> int:
        if context_tokens <= 0 or tasks <= 0:
            return max(1, tasks)
        per_task = context_tokens * self.bytes_per_token
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
        return 1e-9

    reasoning_s = prefill_tokens / model.prefill_tok_per_s + decode_tokens / (
        model.decode_tok_per_s
    )
    k = model.max_streams(prefill_tokens, agents)
    duty = reasoning_s / (reasoning_s + max(0.0, tool_s)) if tool_s > 0 else 1.0
    streams = max(1.0, k * duty)

    total_prefill = agents * rounds * prefill_tokens
    total_decode = agents * rounds * decode_tokens
    device_s = total_prefill / model.aggregate(
        streams, model.prefill_tok_per_s
    ) + total_decode / model.aggregate(streams, model.decode_tok_per_s)
    path_s = rounds * (reasoning_s + max(0.0, tool_s))
    return max(device_s, path_s, 1e-9)
