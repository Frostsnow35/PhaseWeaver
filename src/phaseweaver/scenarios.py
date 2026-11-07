"""Scenario profiles: concrete workloads PhaseWeaver is meant to be used in.

The runtime is generic, but "good" is scenario-dependent: a coding agent wants its
context kept through short tool calls; a research agent fires 30-60 s browser
tools; an orchestrator fans out and joins subagents. Encoding those as profiles
keeps the tuning arguments in one place instead of scattered across experiments.

Each profile references a TOML workload (so numbers stay reproducible) and the
policy that the project recommends for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

WORKLOADS = Path("examples/workloads")


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str
    workload: str
    recommended: str
    why: str

    def path(self) -> Path:
        return WORKLOADS / self.workload


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="coding",
        title="编码 Agent(短工具、长会话)",
        description="Agent 反复执行测试/构建(2 s 级),上下文持续增长,SLO 较宽。",
        workload="w1.toml",
        recommended="auto",
        why="工具短、返回快,RETAIN 通常比 DROP 划算;压力升高时才让出。",
    ),
    Scenario(
        key="research",
        title="浏览/检索 Agent(长工具)",
        description="Agent 调用浏览器/抓取,单次 30-60 s;期间 GPU 完全空闲。",
        workload="w3-queue.toml",
        recommended="auto",
        why="长工具必须让出,否则显存被闲置占用;交接不适用(子任务不共享前缀)。",
    ),
    Scenario(
        key="orchestrator",
        title="多 Agent 编排(扇出/汇合)",
        description="父 Agent 把任务分派给子 Agent,子任务需要父的完整上下文。",
        workload="w9-fanout.toml",
        recommended="handoff",
        why="KV 交接让委托任务零 prefill 起步,扇出场景 p99 显著下降。",
    ),
    Scenario(
        key="delegation",
        title="链式委托(父→子→孙)",
        description="一层层下派的流水线任务,每一跳都需要上一级的上下文。",
        workload="w10-nested.toml",
        recommended="handoff",
        why="交接链的收益随层数放大(每跳都省一次 prefill + 一次重算)。",
    ),
)


def by_key(key: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.key == key:
            return scenario
    raise KeyError(f"unknown scenario {key!r}; available: {[s.key for s in SCENARIOS]}")
