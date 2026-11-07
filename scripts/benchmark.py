"""Performance audit: simulator throughput, scaling, replay cost.

Run: python scripts/benchmark.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseweaver import ToolSpec  # noqa: E402
from phaseweaver.app import mock_caps  # noqa: E402
from phaseweaver.backends.mock import MockBackend  # noqa: E402
from phaseweaver.clock import VirtualClock  # noqa: E402
from phaseweaver.demo import Workload, run_one  # noqa: E402
from phaseweaver.leases import Budget  # noqa: E402
from phaseweaver.policy import YieldOnToolPolicy  # noqa: E402
from phaseweaver.runtime import Runtime, RuntimeConfig  # noqa: E402
from phaseweaver.timeline import build_timeline  # noqa: E402
from phaseweaver.trace import trace_unit  # noqa: E402


def bench_one(agents: int, capacity_gib: float = 2.0, policy: str = "B2", seed: int = 0):  # type: ignore[no-untyped-def]
    w = Workload.load(ROOT / "examples/workloads/w3-queue.toml")
    w.agents = agents
    w.seed = seed
    tracemalloc.start()
    started = time.perf_counter()
    report = asyncio.run(run_one(w, policy, capacity_gib=capacity_gib))
    wall = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return report, wall, peak / 1024 / 1024


async def build_big_journal():  # type: ignore[no-untyped-def]
    clock = VirtualClock()
    clock.start_driver()
    caps = mock_caps(2.0)
    rt = Runtime(
        backend=MockBackend(caps=caps, clock=clock),
        clock=clock,
        budget=Budget(total_bytes=caps.capacity_bytes, headroom_frac=0.05),
        policy=YieldOnToolPolicy(),
        config=RuntimeConfig(run_id="bench"),
    )
    await rt.start()

    async def agent(unit) -> None:  # type: ignore[no-untyped-def]
        prompt = 4096
        for _ in range(3):
            await unit.reason(prompt_tokens=prompt, out_tokens=64)
            await unit.call_tool(
                ToolSpec("t", "python", expected_duration_s=3.0), {"code": "pass"}
            )
            prompt += 64

    tasks = []
    for i in range(12):
        tasks.append(asyncio.create_task(rt.run_task(f"t{i}", agent)))
        await clock.wait(0.25)
    await asyncio.gather(*tasks)
    await rt.shutdown()
    clock.stop_driver()
    return rt


def main() -> int:
    print("## 1. 单次运行开销(8 agents / 2 GiB / B2)")
    report, wall, peak_mb = bench_one(8)
    print(
        f"wall={wall:.2f}s goodput={report.goodput:.4f} "
        f"p99={report.latency_p99_ms / 1000:.1f}s "
        f"peak_kv={report.peak_resident_bytes / 1024 / 1024:.0f}MiB "
        f"process_peak={peak_mb:.1f}MiB"
    )

    print("\n## 2. 规模扩展(B2 / 2 GiB / seed=0)")
    print(f"{'agents':>7}{'wall(s)':>10}{'goodput':>10}{'p99(s)':>9}")
    for agents in (4, 8, 12, 16):
        report, wall, _ = bench_one(agents)
        print(
            f"{agents:>7}{wall:>10.2f}{report.goodput:>10.4f}"
            f"{report.latency_p99_ms / 1000:>9.1f}"
        )

    print("\n## 3. 离线回放成本(12 agents, timeline + 每个 agent 的 trace)")
    rt = asyncio.run(build_big_journal())
    events = list(rt.journal.events())
    started = time.perf_counter()
    timeline = build_timeline(rt.journal)
    traced = 0
    for unit in rt.registry.all_units():
        if unit.kind.value == "agent":
            trace_unit(rt.journal, unit.unit_id, timeline)
            traced += 1
    replay_ms = (time.perf_counter() - started) * 1000
    print(
        f"events={len(events)} spans={len(timeline.spans)} agents_traced={traced} "
        f"replay={replay_ms:.1f}ms "
        f"({len(events) / max(replay_ms, 1e-9) * 1000:.0f} events/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
