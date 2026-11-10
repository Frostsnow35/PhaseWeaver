"""Build the offline playground data by running the *real* simulator.

Run with the package importable (``pip install -e .`` or ``PYTHONPATH=src``):

    python scripts/build_playground_data.py

Every displayed number is an actual ``run_one()`` result, never a JS
reimplementation of the model. Data is injected into site/playground.html
between the PW_DATA markers, so the page stays a single file with no external
requests and works from file://.

Evidence discipline: main grid is [L1] (3 seeds, fixed workload W2 with three
overridden parameters). The prefix panel is [L2] (an assumed upper bound, not
an implemented cache). No [L3] data is produced here.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

from phaseweaver.demo import Workload, run_one
from phaseweaver.journal import EventJournal
from phaseweaver.timeline import build_timeline

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TARGET = ROOT / "site" / "playground.html"
W2 = ROOT / "examples" / "workloads" / "w2.toml"

CAPACITIES = (1.0, 2.0, 4.0, 8.0)
TOOL_P50 = (2.0, 8.0, 20.0)
AGENTS = (4, 8, 12)
POLICIES = ("B0", "B2")
SEEDS = (0, 1, 2)
H_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
METRICS = ("goodput", "p99_ms", "slo_rate", "recompute_tok", "kv_waste_mib_s", "completed")


def _workload(agents: int, tool_p50_s: float, seed: int, hit: float = 0.0) -> Workload:
    workload = Workload.load(W2)
    workload.agents = agents
    workload.tool_p50_s = tool_p50_s
    workload.seed = seed
    workload.prefix_hit_ratio = hit
    return workload


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _row(agents: int, tool: float, capacity: float, policy: str) -> dict[str, float]:
    collected: dict[str, list[float]] = {name: [] for name in METRICS}
    for seed in SEEDS:
        report = asyncio.run(
            run_one(_workload(agents, tool, seed), policy, capacity_gib=capacity)
        )
        summary = report.summary_row()
        for name in METRICS:
            collected[name].append(summary[name])
    return {name: round(_mean(values), 4) for name, values in collected.items()}


def _gantt(agents: int, tool: float, capacity: float, policy: str) -> list[dict[str, object]]:
    """Project one seed's journal into a small Gantt; the journal is discarded."""
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"pw-{uuid.uuid4().hex}.jsonl"
    try:
        asyncio.run(
            run_one(
                _workload(agents, tool, SEEDS[0]),
                policy,
                capacity_gib=capacity,
                journal_path=path,
            )
        )
        timeline = build_timeline(EventJournal.load(path))
    finally:
        path.unlink(missing_ok=True)
    by_unit: dict[str, list[list[object]]] = {}
    for span in timeline.spans:
        if span.kind != "phase":
            continue
        by_unit.setdefault(span.unit_id, []).append(
            [span.name, round(span.start, 2), round(span.end, 2)]
        )
    return [
        {"unit": unit, "spans": spans[:8]}
        for unit, spans in list(by_unit.items())[:5]
    ]


def _prefix_panel() -> list[dict[str, float]]:
    """[L2] null-result panel: even a perfect prefix cache barely moves goodput."""
    out: list[dict[str, float]] = []
    for hit in H_VALUES:
        goodput: list[float] = []
        recompute: list[float] = []
        p99: list[float] = []
        for seed in SEEDS:
            report = asyncio.run(
                run_one(_workload(12, 8.0, seed, hit), "B2", capacity_gib=2.0)
            )
            summary = report.summary_row()
            goodput.append(summary["goodput"])
            recompute.append(summary["recompute_tok"])
            p99.append(summary["p99_ms"] / 1000)
        out.append(
            {
                "h": hit,
                "goodput": round(_mean(goodput), 4),
                "p99_s": round(_mean(p99), 1),
                "recompute_tok": round(_mean(recompute), 0),
            }
        )
    return out


def main() -> int:
    grid: dict[str, dict[str, object]] = {}
    total = len(CAPACITIES) * len(TOOL_P50) * len(AGENTS)
    index = 0
    for capacity in CAPACITIES:
        for tool in TOOL_P50:
            for agents in AGENTS:
                index += 1
                key = f"{capacity:g}|{tool:g}|{agents}"
                entry: dict[str, object] = {
                    "capacity_gib": capacity,
                    "tool_p50_s": tool,
                    "agents": agents,
                    "B0": _row(agents, tool, capacity, "B0"),
                    "B2": _row(agents, tool, capacity, "B2"),
                    "gantt": {
                        "B0": _gantt(agents, tool, capacity, "B0"),
                        "B2": _gantt(agents, tool, capacity, "B2"),
                    },
                }
                grid[key] = entry
                print(f"[{index}/{total}] {key}", flush=True)

    payload = {
        "evidence": "[L1] deterministic simulation / W2 mixed short-long tools / seeds 0,1,2",
        "capacities": list(CAPACITIES),
        "tool_p50": list(TOOL_P50),
        "agents": list(AGENTS),
        "prefix_evidence": "[L2] assumed prefix-cache upper bound, NOT an implemented cache",
        "prefix": _prefix_panel(),
        "grid": grid,
    }
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    template = TARGET.read_text(encoding="utf-8")
    pattern = re.compile(
        r"<!--PW_DATA_BEGIN-->.*?<!--PW_DATA_END-->", re.DOTALL
    )
    block = (
        "<!--PW_DATA_BEGIN-->\n"
        f'<script id="pw-data" type="application/json">{data}</script>\n'
        "<!--PW_DATA_END-->"
    )
    if not pattern.search(template):
        raise SystemExit("site/playground.html: PW_DATA markers not found")
    TARGET.write_text(pattern.sub(lambda _: block, template, count=1), encoding="utf-8")
    print(f"wrote {len(grid)} grid points into {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
