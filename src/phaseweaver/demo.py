"""Workload model and the reproducible simulation runner.

Command-line parsing, table formatting and subcommand dispatch live in
``cli.py``; ``main`` here is a thin shell so ``python -m phaseweaver.demo`` and
the ``phaseweaver`` console script keep working. Every experiment table
produced by the CLI carries an [L1]/[L2] header (spec 12.5).
"""

from __future__ import annotations

import asyncio
import random
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.leases import Budget
from phaseweaver.metrics import MetricsReducer, RunReport
from phaseweaver.policy import Scheduler
from phaseweaver.runtime import Runtime, RuntimeConfig
from phaseweaver.tools.base import ToolSpec
from phaseweaver.types import ForkMode

BYTES_PER_TOKEN = 57344  # Qwen2.5-7B bf16 KV per token
BLOCK_TOKENS = 16


@dataclass
class Workload:
    name: str
    agents: int
    rounds: int
    context_tokens: int
    out_tokens: int
    slo_s: float
    seed: int
    tool_p50_s: float
    tool_sigma: float
    rate_per_s: float
    fanout: int = 0
    sub_rounds: int = 0
    fork_mode: str = "share"
    sub_context_tokens: int = 0
    greedy_factor: float = 1.0
    #: none | handoff | relay  (inherit_prefix kept as a legacy boolean alias)
    inherit_mode: str = "none"
    #: 2 means each subagent also delegates to a grandchild
    nested_depth: int = 1
    #: R6: 0 keeps tools uniform; >0 splits agents into short/long tool groups
    #: (even index: * (1+h), odd index: * (1-h)) so phase alignment has something
    #: to align.
    tool_hetero: float = 0.0
    #: [L2] experimental upper bound: fraction of recompute prefill work that a
    #: shared prefix cache would cover. 0.0 preserves the original model exactly.
    prefix_hit_ratio: float = 0.0

    @classmethod
    def load(cls, path: Path) -> Workload:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        w, t, a = data["workload"], data["tool"], data["arrival"]
        return cls(
            name=str(w["name"]),
            agents=int(w["agents"]),
            rounds=int(w["rounds"]),
            context_tokens=int(w["context_tokens"]),
            out_tokens=int(w["out_tokens"]),
            slo_s=float(w["slo_s"]),
            seed=int(w["seed"]),
            tool_p50_s=float(t["p50_s"]),
            tool_sigma=float(t["sigma"]),
            rate_per_s=float(a["rate_per_s"]),
            fanout=int(w.get("fanout", 0)),
            sub_rounds=int(w.get("sub_rounds", 0)),
            fork_mode=str(w.get("fork_mode", "share")),
            sub_context_tokens=int(w.get("sub_context_tokens", 0)),
            greedy_factor=float(w.get("greedy_factor", 1.0)),
            inherit_mode=str(
                w.get("inherit_mode")
                or ("handoff" if w.get("inherit_prefix") in (True, "true") else "none")
            ),
            nested_depth=int(w.get("nested_depth", 1)),
            tool_hetero=float(t.get("hetero", 0.0)),
            prefix_hit_ratio=float(w.get("prefix_hit_ratio", 0.0)),
        )


def caps(capacity_gib: float = 8.0) -> BackendCaps:
    return BackendCaps(
        capacity_bytes=int(capacity_gib * 1024**3),
        block_tokens=BLOCK_TOKENS,
        bytes_per_token=BYTES_PER_TOKEN,
        prefill_tok_per_s=3200,
        decode_tok_per_s=45,
        mem_bw_bps=int(12 * 1024**3),
        supports_offload=True,
    )


def build_policy(name: str) -> Scheduler:
    """Single source of truth: policy names are owned by the facade, not the CLI.

    (Keeping a second copy here is how `align2` silently became "unknown policy".)
    """
    from phaseweaver.app import build_policy as facade_build_policy

    return facade_build_policy(name)


async def run_one(
    workload: Workload,
    policy_name: str,
    *,
    capacity_gib: float = 8.0,
    journal_path: Path | None = None,
    backend_caps: BackendCaps | None = None,
) -> RunReport:
    """Run one workload. ``backend_caps`` overrides the mock defaults so a
    real-GPU calibration (``scripts/probe_calibration.py``) can re-run the
    [L2] gates without editing the simulator. ``None`` keeps byte-identical
    behaviour for every existing caller.
    """
    clock = VirtualClock()
    clock.start_driver()
    backend_caps = backend_caps or caps(capacity_gib)
    backend = MockBackend(
        caps=backend_caps, clock=clock, prefix_hit_ratio=workload.prefix_hit_ratio
    )
    runtime = Runtime(
        backend=backend,
        clock=clock,
        budget=Budget(total_bytes=backend_caps.capacity_bytes, headroom_frac=0.05),
        policy=build_policy(policy_name),
        config=RuntimeConfig(run_id=f"{workload.name}-{policy_name}", journal_path=journal_path),
    )
    await runtime.start()
    rng = random.Random(workload.seed)
    tools = [
        ToolSpec(
            f"tool-{i}",
            "python",
            expected_duration_s=max(
                0.1, rng.lognormvariate(0.0, workload.tool_sigma) * workload.tool_p50_s
            ),
        )
        for i in range(workload.rounds)
    ]

    async def child(handle: Any, greedy: bool = False) -> None:
        base = workload.sub_context_tokens or workload.context_tokens // 2
        base = int(base * (workload.greedy_factor if greedy else 1.0))
        # Same conversation length in both arms; only the sharing differs, otherwise
        # the comparison would just measure a smaller prompt.
        prompt = workload.context_tokens + base
        for _ in range(workload.sub_rounds):
            await handle.reason(prompt_tokens=prompt, out_tokens=workload.out_tokens)
            await handle.call_tool(
                ToolSpec("sub-tool", "python", expected_duration_s=workload.tool_p50_s),
                {"code": "pass"},
            )
            prompt += workload.out_tokens
        if workload.nested_depth >= 2:
            await handle.fork(
                "grandchild", _grandchild, inherit=workload.inherit_mode
            )

    async def _grandchild(handle: Any) -> None:
        await handle.reason(
            prompt_tokens=workload.context_tokens // 2, out_tokens=workload.out_tokens
        )

    async def agent(handle: Any, specs: list[ToolSpec] | None = None) -> None:
        prompt = workload.context_tokens
        for spec in specs or tools:
            await handle.reason(prompt_tokens=prompt, out_tokens=workload.out_tokens)
            await handle.call_tool(spec, {"code": "pass"})
            prompt += workload.out_tokens  # a conversation grows every round
        if workload.fanout > 0:
            # Fork a group of subagents: the parent parks and waits (WAITING),
            # children share the parent's budget (SHARE) or get a hard cap (ISOLATE).
            mode = ForkMode.ISOLATE if workload.fork_mode == "isolate" else ForkMode.SHARE
            if workload.inherit_mode == "relay":
                # Relay: children run one after another, each inheriting the KV from
                # the previous one through the parent - every delegate starts warm.
                for i in range(workload.fanout):
                    await handle.fork(
                        f"sub-{i}",
                        lambda h, g=(i == 0): child(h, g),
                        mode=mode,
                        inherit=workload.inherit_mode,
                    )
            else:
                await asyncio.gather(
                    *(
                        handle.fork(
                            f"sub-{i}",
                            lambda h, g=(i == 0): child(h, g),
                            mode=mode,
                            inherit=workload.inherit_mode,
                        )
                        for i in range(workload.fanout)
                    )
                )
            await handle.reason(prompt_tokens=prompt, out_tokens=workload.out_tokens)

    pending: list[asyncio.Task[Any]] = []
    for i in range(workload.agents):
        if workload.rate_per_s > 0:
            await clock.wait(rng.expovariate(workload.rate_per_s))
        if workload.tool_hetero > 0:
            scale = 1.0 + workload.tool_hetero if i % 2 == 0 else max(
                0.1, 1.0 - workload.tool_hetero
            )
            specs = [
                ToolSpec(
                    spec.name,
                    spec.executor,
                    expected_duration_s=(spec.expected_duration_s or workload.tool_p50_s)
                    * scale,
                )
                for spec in tools
            ]

            async def tuned(handle: Any, chosen: list[ToolSpec] = specs) -> None:
                await agent(handle, chosen)

            job = runtime.run_task(f"task-{i}", tuned, slo_s=workload.slo_s)
        else:
            job = runtime.run_task(f"task-{i}", agent, slo_s=workload.slo_s)
        pending.append(asyncio.create_task(job))
    await asyncio.gather(*pending)
    await runtime.shutdown()
    clock.stop_driver()
    return MetricsReducer.replay(runtime.journal)


def main(argv: list[str] | None = None) -> int:
    """Entry point kept for ``python -m phaseweaver.demo`` and the console script."""
    from phaseweaver.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
