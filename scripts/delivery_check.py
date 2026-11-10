"""交付闸门:一条命令回答"这个仓库现在能不能交付"。

这不是 CI 的替代品,而是 **本地交付标准的可执行版本**:CI 保证"不坏",
这里保证"达标"(行为、性能、可观测性都有数值门槛)。

用法:
    python scripts/delivery_check.py            # 全量(约 2-3 分钟)
    python scripts/delivery_check.py --quick    # 跳过性能闸门
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


def run(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()
    return proc.returncode == 0, tail[-1] if tail else ""


def gate_tests() -> Gate:
    ok, last = run([sys.executable, "-m", "pytest", "-q"])
    return Gate("tests", ok, last)


def gate_lint() -> Gate:
    """Prefer the ruff executable, then the module form.

    Some uv environments install the ``ruff`` wheel without a console script in
    the venv, so ``python -m ruff`` cannot locate its binary even though ruff is
    on PATH. Trying both keeps the gate honest on those machines.
    """
    last = "ruff not found"
    for cmd in (["ruff", "check", "."], [sys.executable, "-m", "ruff", "check", "."]):
        try:
            ok, output = run(cmd)
        except OSError:
            continue
        last = output
        if ok:
            return Gate("lint", True, last)
    return Gate("lint", False, last)


def gate_types() -> Gate:
    ok_linux, last = run([sys.executable, "-m", "mypy", "--platform", "linux", "src/phaseweaver"])
    ok_win, _ = run([sys.executable, "-m", "mypy", "--platform", "win32", "src/phaseweaver"])
    return Gate("types(linux+win32)", ok_linux and ok_win, last)


def gate_determinism() -> Gate:
    from phaseweaver.demo import Workload, run_one

    w = Workload.load(ROOT / "examples/workloads/w3-queue.toml")
    w.agents = 6
    a = asyncio.run(run_one(w, "B2", capacity_gib=2.0))
    b = asyncio.run(run_one(w, "B2", capacity_gib=2.0))
    ok = a.summary_row() == b.summary_row()
    return Gate("determinism", ok, "same seed => identical metrics" if ok else "runs differ")


def gate_performance() -> Gate:
    """B2 must not be worse than B0 on the queueing scenario (SLO, p99)."""
    from phaseweaver.demo import Workload, run_one

    rows = {}
    for policy in ("B0", "B2"):
        slo = []
        p99 = []
        goodput = []
        for seed in range(2):
            w = Workload.load(ROOT / "examples/workloads/w3-queue.toml")
            w.seed = seed
            report = asyncio.run(run_one(w, policy, capacity_gib=2.0))
            slo.append(report.slo_attainment_rate)
            p99.append(report.latency_p99_ms)
            goodput.append(report.goodput)
        rows[policy] = (
            sum(slo) / len(slo),
            sum(p99) / len(p99),
            sum(goodput) / len(goodput),
        )
    b0, b2 = rows["B0"], rows["B2"]
    ok = b2[0] >= b0[0] - 1e-9 and b2[1] <= b0[1] * 1.02
    detail = (
        f"B0 slo={b0[0]:.3f} p99={b0[1] / 1000:.1f}s | "
        f"B2 slo={b2[0]:.3f} p99={b2[1] / 1000:.1f}s"
    )
    return Gate("performance(B2>=B0)", ok, detail)


def gate_observability() -> Gate:
    from phaseweaver.demo import Workload
    from phaseweaver.timeline import build_timeline
    from phaseweaver.trace import trace_unit

    w = Workload.load(ROOT / "examples/workloads/w3-queue.toml")
    w.agents = 4
    report_journal: object | None = None

    async def scenario():  # type: ignore[no-untyped-def]
        nonlocal report_journal
        from phaseweaver.app import mock_caps
        from phaseweaver.backends.mock import MockBackend
        from phaseweaver.clock import VirtualClock
        from phaseweaver.leases import Budget
        from phaseweaver.policy import YieldOnToolPolicy
        from phaseweaver.runtime import Runtime, RuntimeConfig

        clock = VirtualClock()
        clock.start_driver()
        caps = mock_caps(2.0)
        rt = Runtime(
            backend=MockBackend(caps=caps, clock=clock),
            clock=clock,
            budget=Budget(total_bytes=caps.capacity_bytes, headroom_frac=0.05),
            policy=YieldOnToolPolicy(),
            config=RuntimeConfig(run_id="obs"),
        )
        await rt.start()

        async def agent(unit) -> None:  # type: ignore[no-untyped-def]
            await unit.reason(prompt_tokens=2048, out_tokens=32)

        await rt.run_task("t", agent)
        await rt.shutdown()
        clock.stop_driver()
        report_journal = rt.journal

    asyncio.run(scenario())
    assert report_journal is not None
    started = time.perf_counter()
    timeline = build_timeline(report_journal)  # type: ignore[arg-type]
    trace = trace_unit(report_journal, "obs/root-0-0", timeline)  # type: ignore[arg-type]
    elapsed_ms = (time.perf_counter() - started) * 1000
    ok = bool(timeline.spans) and trace.latency_s > 0 and elapsed_ms < 200
    return Gate("observability", ok, f"{len(timeline.spans)} spans, replay {elapsed_ms:.1f}ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip the performance gate")
    args = parser.parse_args()

    gates: list[Gate] = [
        gate_tests(),
        gate_lint(),
        gate_types(),
        gate_determinism(),
        gate_observability(),
    ]
    if not args.quick:
        gates.append(gate_performance())

    width = max(len(g.name) for g in gates)
    print("\n交付闸门")
    print("-" * 70)
    for gate in gates:
        mark = "PASS" if gate.passed else "FAIL"
        print(f"{gate.name:<{width}}  {mark}  {gate.detail}")
    print("-" * 70)
    failed = [g for g in gates if not g.passed]
    print(f"{len(gates) - len(failed)}/{len(gates)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
