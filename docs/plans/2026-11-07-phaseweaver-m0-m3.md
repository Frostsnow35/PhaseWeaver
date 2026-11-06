# PhaseWeaver 第一阶段实施计划(M0–M3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本机(无 GPU)交付一个可运行的 PhaseWeaver 垂直切片:执行单元树 + 双状态机 + 租约/处置 + 全局 SLO 调度 + mock 仿真 + 指标与实验矩阵。

**Architecture:** 进程内 asyncio 运行时(ADR-0004)。阶段(`Phase`)与推理上下文(`ContextState`)正交建模(ADR-0001);租约表达"处置权 + 成本"(ADR-0002);一切状态变更写入 append-only 事件日志,指标由日志重放得出(ADR-0003);调度目标为 SLO 达成率且 p99 不劣化(ADR-0005)。

**Tech Stack:** Python 3.12、标准库 asyncio/heapq/dataclasses/tomllib、pytest、hypothesis、ruff、mypy、uv。运行时核心零第三方依赖。

---

## 0. 环境与硬约束

- 工作目录:`E:\自我学习项目\PhaseWeaver`(已 `git init`,已有两条 docs 提交)。
- 所有命令用 `uv run ...` 执行,保证同一解释器。
- **Windows 硬约束**:`asyncio.create_subprocess_*` 必须显式传 `creationflags`(`CREATE_NEW_PROCESS_GROUP`)且不传 `start_new_session`;POSIX 分支反之。`ProcessReaper` 必须两平台都实现,即使当前只在 Windows 上跑。
- **模拟器硬约束**:禁止在 mock 仿真路径里调用 `time.sleep` 或 `asyncio.sleep(正的秒数)`;一律 `await clock.wait(dt)`。
- 提交粒度:每个 Task 结束提交一次,前缀 `feat:` / `test:` / `docs:` / `chore:`。

### 每个 Task 的固定收尾命令

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src/phaseweaver
git add -A && git commit -m "<type>: <scope>"
```

---

## 1. 文件结构总图(先定边界,再写代码)

```
pyproject.toml
src/phaseweaver/
  __init__.py        导出 Runtime / Phase / Disposition / Budget
  errors.py          异常层级
  types.py           Phase/ContextState/UnitKind/LeaseKind/Disposition/ForkMode/InferenceContext
  phases.py          两张转移表 + 校验函数(唯一权威)
  clock.py           Clock 协议 / RealClock / VirtualClock(离散事件驱动)
  journal.py         Event / EventJournal(JSONL + 内存索引 + digest)
  units.py           ExecutionUnit / UnitRegistry(fork/join/transition/cancel 钩子)
  leases.py          Budget / Lease / LeaseBroker
  invariant.py       InvariantChecker(I1–I4, I6, I8)
  policy.py          ClusterSnapshot / Scheduler 协议 / PinAlways / TtlEvict / YieldOnTool
  scheduler.py       SystemScheduler(全局 SLO + 防抖 + DRR)
  backends/base.py   BackendCaps / Handle / GenStats / ParkResult / RestoreResult / 协议
  backends/mock.py   MockBackend(块分配 + 虚拟时间 + 引擎事件)
  backends/vllm.py   (M5,本计划只留 stub 文件)
  backends/openai_compat.py (M5,本计划只留 stub 文件)
  tools/base.py      ToolSpec / ToolResult / ToolExecutor 协议
  tools/reaper.py    跨平台进程树回收
  tools/python_exec.py / shell_exec.py / browser_exec.py
  tools/supervisor.py ToolSupervisor(超时/取消/结果归一)
  metrics.py         RunReport / MetricsReducer(从 journal 重放)
  runtime.py         Runtime / AgentHandle(接线:阶段 ↔ 处置 ↔ 恢复)
  oracle.py          B4 离线最优上界(只被实验代码 import)
  demo.py            CLI:compare / sweep / seeds / optimal
tests/
  unit/ property/ integration/
examples/workloads/*.toml
```

模块纪律:`phases.py`、`leases.py`、`invariant.py`、`metrics.py` 不得 import `runtime.py` / `scheduler.py`;`oracle.py` 不得被 `runtime.py` import。以 `tests/unit/test_import_graph.py` 固化。

---

## Task 1: 项目骨架与工具链

**Files:**
- Create: `pyproject.toml`, `src/phaseweaver/__init__.py`, `tests/__init__.py`, `tests/unit/test_smoke.py`

- [ ] **Step 1: 写 `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "phaseweaver"
version = "0.1.0"
description = "Phase-aware agent runtime for GPU resource coordination"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8", "hypothesis>=6", "ruff>=0.6", "mypy>=1.11"]

[tool.hatch.build.targets.wheel]
packages = ["src/phaseweaver"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "TID"]

[tool.mypy]
python_version = "3.11"
strict = true
files = ["src/phaseweaver"]
```

- [ ] **Step 2: 写 `src/phaseweaver/__init__.py`**

```python
"""PhaseWeaver: phase-aware agent runtime for GPU resource coordination."""

__version__ = "0.1.0"
__all__ = ["__version__"]
```

- [ ] **Step 3: 写冒烟测试 `tests/unit/test_smoke.py`**

```python
import phaseweaver


def test_version() -> None:
    assert phaseweaver.__version__ == "0.1.0"
```

- [ ] **Step 4: 安装并跑通**

Run: `uv sync --extra dev && uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "chore: bootstrap package, tooling and smoke test"
```

---

## Task 2: 核心类型与异常

**Files:**
- Create: `src/phaseweaver/errors.py`, `src/phaseweaver/types.py`
- Test: `tests/unit/test_types.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_types.py`**

```python
from phaseweaver.types import ContextState, InferenceContext, Phase


def test_terminal_phases() -> None:
    assert Phase.COMPLETED.is_terminal()
    assert Phase.FAILED.is_terminal()
    assert Phase.CANCELLED.is_terminal()
    assert not Phase.ACTING.is_terminal()


def test_context_defaults() -> None:
    ctx = InferenceContext(context_id="c1")
    assert ctx.state is ContextState.NONE
    assert ctx.resident_bytes == 0


def test_context_occupies_gpu() -> None:
    ctx = InferenceContext(context_id="c1", state=ContextState.ACTIVE, resident_bytes=4096)
    assert ctx.occupies_gpu
    ctx.state = ContextState.OFFLOADED
    assert not ctx.occupies_gpu
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/unit/test_types.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.types`

- [ ] **Step 3: 写 `src/phaseweaver/errors.py`**

```python
"""Exception hierarchy. 所有可预期的运行时错误都继承 PhaseWeaverError。"""


class PhaseWeaverError(Exception):
    """Base class."""


class IllegalTransition(PhaseWeaverError):
    def __init__(self, kind: str, src: str, dst: str) -> None:
        super().__init__(f"illegal {kind} transition: {src} -> {dst}")
        self.kind = kind
        self.src = src
        self.dst = dst


class AdmissionRejected(PhaseWeaverError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"admission rejected: {reason}")
        self.reason = reason


class BudgetExceeded(PhaseWeaverError):
    pass


class JoinBeforeTermination(PhaseWeaverError):
    pass


class ToolTimeout(PhaseWeaverError):
    pass


class ReapIncomplete(PhaseWeaverError):
    pass


class EngineOutOfMemory(PhaseWeaverError):
    """Runtime 的记账出现 bug:租约允许但引擎装不下。"""


class ConfigError(PhaseWeaverError):
    pass
```

- [ ] **Step 4: 写 `src/phaseweaver/types.py`**

```python
"""核心值类型。此模块不得 import 其它内部模块(避免环)。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

RunId = str
UnitId = str
ContextId = str
LeaseId = str


class Phase(StrEnum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    REASONING = "REASONING"
    ACTING = "ACTING"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def is_terminal(self) -> bool:
        return self in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED)


class ContextState(StrEnum):
    NONE = "NONE"
    ACTIVE = "ACTIVE"
    RETAINED = "RETAINED"
    OFFLOADED = "OFFLOADED"
    DROPPED = "DROPPED"
    RELEASED = "RELEASED"


class UnitKind(StrEnum):
    AGENT = "agent"
    SUBAGENT = "subagent"
    TOOL_CALL = "tool_call"


class LeaseKind(StrEnum):
    KV_BYTES = "kv_bytes"
    OFFLOAD_BYTES = "offload_bytes"
    ENGINE_SLOT = "engine_slot"
    TOOL_SLOT = "tool_slot"


class Disposition(StrEnum):
    RETAIN = "retain"
    OFFLOAD = "offload"
    DROP = "drop"
    RELEASE = "release"


class ForkMode(StrEnum):
    SHARE = "share"
    ISOLATE = "isolate"


@dataclass
class InferenceContext:
    context_id: ContextId
    state: ContextState = ContextState.NONE
    resident_tokens: int = 0
    resident_bytes: int = 0
    offload_bytes: int = 0
    updated_seq: int | None = None

    @property
    def occupies_gpu(self) -> bool:
        return self.state in (ContextState.ACTIVE, ContextState.RETAINED)

    @property
    def needs_recompute(self) -> bool:
        return self.state is ContextState.DROPPED


@dataclass
class UnitAggregate:
    """join 时的合并记账。"""

    tasks_completed: int = 0
    recompute_tokens: int = 0
    park_count: int = 0
    child_ids: list[UnitId] = field(default_factory=list)
```

- [ ] **Step 5: 运行测试**

Run: `uv run pytest tests/unit/test_types.py -q`
Expected: `3 passed`

- [ ] **Step 6: 提交**

```bash
git add -A && git commit -m "feat(types): phase/context enums, inference context and error hierarchy"
```

---

## Task 3: 双状态机(阶段 + 上下文)

**Files:**
- Create: `src/phaseweaver/phases.py`
- Test: `tests/unit/test_phases.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_phases.py`**

```python
import pytest

from phaseweaver.errors import IllegalTransition
from phaseweaver.phases import (
    CONTEXT_TRANSITIONS,
    PHASE_TRANSITIONS,
    check_context_transition,
    check_phase_transition,
)
from phaseweaver.types import ContextState, Phase


def test_phase_legal_transition() -> None:
    check_phase_transition(Phase.ACTING, Phase.REASONING)


def test_phase_illegal_transition() -> None:
    with pytest.raises(IllegalTransition):
        check_phase_transition(Phase.COMPLETED, Phase.ACTING)


def test_all_phases_in_table() -> None:
    for phase in Phase:
        assert phase in PHASE_TRANSITIONS


def test_terminal_phases_have_no_exit() -> None:
    for phase in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        assert PHASE_TRANSITIONS[phase] == frozenset()


def test_context_transition_dropped_can_restore() -> None:
    check_context_transition(ContextState.DROPPED, ContextState.ACTIVE)


def test_context_released_is_final() -> None:
    assert CONTEXT_TRANSITIONS[ContextState.RELEASED] == frozenset()
    with pytest.raises(IllegalTransition):
        check_context_transition(ContextState.RELEASED, ContextState.ACTIVE)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/unit/test_phases.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.phases`

- [ ] **Step 3: 写 `src/phaseweaver/phases.py`**

```python
"""两张转移表:Phase(执行阶段)与 ContextState(推理上下文)。

两者正交(ADR-0001),本模块是它们唯一的权威定义。
"""

from __future__ import annotations

from typing import Final

from phaseweaver.errors import IllegalTransition
from phaseweaver.types import ContextState, Phase, UnitKind

PHASE_TRANSITIONS: Final[dict[Phase, frozenset[Phase]]] = {
    # PENDING -> ACTING 专供 TOOL_CALL 单元(工具不经推理);UnitRegistry 会校验 kind
    Phase.PENDING: frozenset({Phase.QUEUED, Phase.ACTING, Phase.CANCELLED}),
    Phase.QUEUED: frozenset({Phase.REASONING, Phase.CANCELLED}),
    Phase.REASONING: frozenset(
        {
            Phase.ACTING,
            Phase.WAITING,
            Phase.COMPLETED,
            Phase.FAILED,
            Phase.CANCELLED,
        }
    ),
    Phase.ACTING: frozenset({Phase.REASONING, Phase.FAILED, Phase.CANCELLED}),
    Phase.WAITING: frozenset({Phase.REASONING, Phase.FAILED, Phase.CANCELLED}),
    Phase.COMPLETED: frozenset(),
    Phase.FAILED: frozenset(),
    Phase.CANCELLED: frozenset(),
}

CONTEXT_TRANSITIONS: Final[dict[ContextState, frozenset[ContextState]]] = {
    ContextState.NONE: frozenset({ContextState.ACTIVE, ContextState.RELEASED}),
    ContextState.ACTIVE: frozenset(
        {
            ContextState.ACTIVE,
            ContextState.RETAINED,
            ContextState.OFFLOADED,
            ContextState.DROPPED,
            ContextState.RELEASED,
        }
    ),
    ContextState.RETAINED: frozenset(
        {
            ContextState.ACTIVE,
            ContextState.OFFLOADED,
            ContextState.DROPPED,
            ContextState.RELEASED,
        }
    ),
    ContextState.OFFLOADED: frozenset({ContextState.ACTIVE, ContextState.RELEASED}),
    ContextState.DROPPED: frozenset({ContextState.ACTIVE, ContextState.RELEASED}),
    ContextState.RELEASED: frozenset(),
}

#: I1:这些阶段不得持有 ACTIVE 上下文
NON_GPU_PHASES: Final[frozenset[Phase]] = frozenset(
    {Phase.ACTING, Phase.WAITING, Phase.QUEUED, Phase.PENDING}
)


def check_phase_transition(src: Phase, dst: Phase, *, kind: UnitKind | None = None) -> None:
    if dst not in PHASE_TRANSITIONS[src]:
        raise IllegalTransition("phase", str(src), str(dst))
    if src is Phase.PENDING and dst is Phase.ACTING and kind is not UnitKind.TOOL_CALL:
        raise IllegalTransition("phase(kind=agent)", str(src), str(dst))


def check_context_transition(src: ContextState, dst: ContextState) -> None:
    if dst not in CONTEXT_TRANSITIONS[src]:
        raise IllegalTransition("context", str(src), str(dst))
```

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/unit/test_phases.py -q`
Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat(phases): dual state machines with authoritative transition tables"
```

---

## Task 4: 事件日志

**Files:**
- Create: `src/phaseweaver/clock.py`, `src/phaseweaver/journal.py`
- Test: `tests/unit/test_journal.py`

- [ ] **Step 1: 写 `src/phaseweaver/clock.py`(先给可用实现,虚拟时钟在 Task 9 完善)**

```python
"""时钟抽象。仿真必须走 VirtualClock,禁止直接 sleep(见计划 §0)。"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...
    async def wait(self, dt: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return asyncio.get_running_loop().time()

    async def wait(self, dt: float) -> None:
        if dt > 0:
            await asyncio.sleep(dt)


class VirtualClock:
    """离散事件虚拟时钟。

    时间只在"没有可运行协程"时跳到下一个定时器;相同 seed 下事件序列确定。
    """

    def __init__(self, start: float = 0.0, idle_rounds: int = 32) -> None:
        self._now = start
        self._heap: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self._idle_rounds = idle_rounds
        self._wakeups = 0
        self._idle = 0
        self._last_wakeups = -1
        self._driver: asyncio.Task[None] | None = None

    def now(self) -> float:
        return self._now

    async def wait(self, dt: float) -> None:
        if dt <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (self._now + dt, next(self._seq), fut))
        self._wakeups += 1
        await fut

    def start_driver(self) -> None:
        if self._driver is None:
            self._driver = asyncio.create_task(self._drive(), name="virtual-clock")

    async def _drive(self) -> None:
        while True:
            await asyncio.sleep(0)
            if not self._heap:
                self._last_wakeups = self._wakeups
                self._idle = 0
                continue
            if self._wakeups == self._last_wakeups:
                self._idle += 1
            else:
                self._idle = 0
                self._last_wakeups = self._wakeups
            if self._idle >= self._idle_rounds:
                self._jump()
                self._idle = 0

    def _jump(self) -> None:
        if not self._heap:
            return
        deadline = self._heap[0][0]
        self._now = deadline
        while self._heap and self._heap[0][0] <= deadline:
            _, _, fut = heapq.heappop(self._heap)
            if not fut.done():
                fut.set_result(None)
            self._wakeups += 1
```

- [ ] **Step 2: 写失败测试 `tests/unit/test_journal.py`**

```python
import asyncio
import json

from phaseweaver.clock import VirtualClock
from phaseweaver.journal import EventJournal


def test_append_assigns_monotonic_seq(tmp_path) -> None:
    clock = VirtualClock()
    j = EventJournal(run_id="r1", clock=clock)
    e1 = j.append("run_started", payload={"x": 1})
    e2 = j.append("unit_created", unit_id="r1/root")
    assert (e1.seq, e2.seq) == (1, 2)
    assert e2.unit_id == "r1/root"


def test_jsonl_file_is_valid(tmp_path) -> None:
    path = tmp_path / "ev.jsonl"
    j = EventJournal(run_id="r1", clock=VirtualClock(), path=path)
    j.append("run_started")
    j.append("run_finished", payload={"ok": True})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["payload"] == {"ok": True}


def test_digest_is_stable_for_same_input() -> None:
    def build() -> EventJournal:
        j = EventJournal(run_id="r1", clock=VirtualClock())
        j.append("run_started")
        j.append("phase_transitioned", unit_id="u", phase_from="QUEUED", phase_to="REASONING")
        return j

    assert build().digest() == build().digest()


def test_virtual_clock_ordering() -> None:
    async def scenario() -> list[float]:
        clock = VirtualClock()
        clock.start_driver()
        marks: list[float] = []

        async def worker(dt: float) -> None:
            await clock.wait(dt)
            marks.append(clock.now())

        await asyncio.gather(worker(2.0), worker(1.0), worker(3.0))
        return marks

    assert asyncio.run(scenario()) == [1.0, 2.0, 3.0]
```

- [ ] **Step 3: 运行确认失败**

Run: `uv run pytest tests/unit/test_journal.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.journal`

- [ ] **Step 4: 写 `src/phaseweaver/journal.py`**

```python
"""Append-only 事件日志:系统的唯一事实来源(ADR-0003)。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phaseweaver.clock import Clock


@dataclass(frozen=True)
class Event:
    seq: int
    ts: float
    run_id: str
    event: str
    unit_id: str | None = None
    phase_from: str | None = None
    phase_to: str | None = None
    cause_seq: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "ts": self.ts,
                "run_id": self.run_id,
                "event": self.event,
                "unit_id": self.unit_id,
                "phase_from": self.phase_from,
                "phase_to": self.phase_to,
                "cause_seq": self.cause_seq,
                "payload": self.payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class EventJournal:
    def __init__(self, run_id: str, clock: Clock, path: Path | None = None) -> None:
        self.run_id = run_id
        self._clock = clock
        self._path = path
        self._events: list[Event] = []
        self._seq = 0
        self._fh = path.open("w", encoding="utf-8", newline="\n") if path else None

    def append(
        self,
        event: str,
        *,
        unit_id: str | None = None,
        phase_from: str | None = None,
        phase_to: str | None = None,
        cause_seq: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        self._seq += 1
        ev = Event(
            seq=self._seq,
            ts=self._clock.now(),
            run_id=self.run_id,
            event=event,
            unit_id=unit_id,
            phase_from=phase_from,
            phase_to=phase_to,
            cause_seq=cause_seq,
            payload=payload or {},
        )
        self._events.append(ev)
        if self._fh is not None:
            self._fh.write(ev.to_json() + "\n")
            self._fh.flush()
        return ev

    def events(self) -> Iterator[Event]:
        return iter(self._events)

    def by_type(self, event: str) -> list[Event]:
        return [e for e in self._events if e.event == event]

    def digest(self) -> str:
        h = hashlib.sha256()
        for e in self._events:
            h.update(e.to_json().encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
```

- [ ] **Step 5: 运行测试**

Run: `uv run pytest tests/unit/test_journal.py -q`
Expected: `4 passed`

> 若 `test_virtual_clock_ordering` 偶发失败,把 `VirtualClock(idle_rounds=32)` 调大到 64,并在提交信息里记录该参数。

- [ ] **Step 6: 提交**

```bash
git add -A && git commit -m "feat(journal): append-only event journal with digest and virtual clock"
```

---

## Task 5: 单元树与阶段转移(I6)

**Files:**
- Create: `src/phaseweaver/units.py`
- Test: `tests/unit/test_units.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_units.py`**

```python
import pytest

from phaseweaver.clock import VirtualClock
from phaseweaver.errors import IllegalTransition, JoinBeforeTermination
from phaseweaver.journal import EventJournal
from phaseweaver.types import ForkMode, Phase, UnitKind
from phaseweaver.units import UnitRegistry


def make_registry() -> tuple[UnitRegistry, EventJournal]:
    clock = VirtualClock()
    journal = EventJournal(run_id="r1", clock=clock)
    return UnitRegistry(journal=journal, clock=clock), journal


def test_create_root_and_fork() -> None:
    reg, _ = make_registry()
    root = reg.create_root("solve")
    child = reg.fork(root.unit_id, "sub-0", mode=ForkMode.SHARE)
    assert root.kind is UnitKind.AGENT
    assert child.kind is UnitKind.SUBAGENT
    assert child.unit_id.startswith(root.unit_id + "/")
    assert child.unit_id in reg.descendants(root.unit_id)


def test_transition_writes_exactly_one_event() -> None:
    reg, journal = make_registry()
    root = reg.create_root("solve")
    reg.transition(root.unit_id, Phase.QUEUED)
    transitions = journal.by_type("phase_transitioned")
    assert len(transitions) == 1
    assert transitions[0].payload["from"] == "PENDING"


def test_illegal_transition_rejected_and_logged() -> None:
    reg, journal = make_registry()
    root = reg.create_root("solve")
    reg.transition(root.unit_id, Phase.QUEUED)
    with pytest.raises(IllegalTransition):
        reg.transition(root.unit_id, Phase.ACTING)
    assert len(journal.by_type("transition_rejected")) == 1


def test_join_requires_terminal_child() -> None:
    reg, _ = make_registry()
    root = reg.create_root("solve")
    child = reg.fork(root.unit_id, "sub-0")
    with pytest.raises(JoinBeforeTermination):
        reg.join(child.unit_id)


def test_parent_terminal_implies_children_terminal() -> None:
    reg, journal = make_registry()
    root = reg.create_root("solve")
    child = reg.fork(root.unit_id, "sub-0")
    reg.transition(root.unit_id, Phase.QUEUED)
    reg.transition(root.unit_id, Phase.REASONING)
    assert not reg.children_all_terminal(root.unit_id)
    reg.transition(child.unit_id, Phase.QUEUED)
    reg.transition(child.unit_id, Phase.REASONING)
    reg.transition(child.unit_id, Phase.COMPLETED)
    assert reg.children_all_terminal(root.unit_id)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/unit/test_units.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.units`

- [ ] **Step 3: 写 `src/phaseweaver/units.py`**

```python
"""执行单元树:fork / join / 树形取消 / 阶段转移。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from phaseweaver.clock import Clock
from phaseweaver.errors import IllegalTransition, JoinBeforeTermination
from phaseweaver.journal import EventJournal
from phaseweaver.phases import check_phase_transition
from phaseweaver.types import (
    ForkMode,
    InferenceContext,
    Phase,
    UnitAggregate,
    UnitId,
    UnitKind,
)

CancelHook = Callable[[UnitId], Awaitable[None]]


@dataclass
class ExecutionUnit:
    unit_id: UnitId
    run_id: str
    kind: UnitKind
    label: str
    parent_id: UnitId | None = None
    mode: ForkMode = ForkMode.SHARE
    phase: Phase = Phase.PENDING
    context: InferenceContext | None = None
    priority: int = 5
    deadline: float | None = None
    arrival_ts: float = 0.0
    children: list[UnitId] = field(default_factory=list)
    aggregate: UnitAggregate = field(default_factory=UnitAggregate)

    @property
    def is_terminal(self) -> bool:
        return self.phase.is_terminal()


class UnitRegistry:
    def __init__(self, journal: EventJournal, clock: Clock) -> None:
        self._journal = journal
        self._clock = clock
        self._units: dict[UnitId, ExecutionUnit] = {}
        self._cancel_hooks: list[CancelHook] = []
        self._lock = asyncio.Lock()

    # --- 创建 -------------------------------------------------------------
    def create_root(
        self, label: str, *, priority: int = 5, deadline: float | None = None
    ) -> ExecutionUnit:
        unit_id = f"{self._journal.run_id}/root"
        unit = ExecutionUnit(
            unit_id=unit_id,
            run_id=self._journal.run_id,
            kind=UnitKind.AGENT,
            label=label,
            priority=priority,
            deadline=deadline,
            arrival_ts=self._clock.now(),
        )
        self._units[unit_id] = unit
        self._journal.append(
            "unit_created",
            unit_id=unit_id,
            payload={
                "kind": unit.kind.value,
                "label": label,
                "arrival_ts": unit.arrival_ts,
                "deadline": deadline,
                "priority": priority,
            },
        )
        return unit

    def fork(
        self,
        parent_id: UnitId,
        label: str,
        *,
        mode: ForkMode = ForkMode.SHARE,
        priority: int | None = None,
    ) -> ExecutionUnit:
        parent = self._units[parent_id]
        index = len(parent.children)
        unit_id = f"{parent_id}/sub-{index}"
        unit = ExecutionUnit(
            unit_id=unit_id,
            run_id=parent.run_id,
            kind=UnitKind.SUBAGENT,
            label=label,
            parent_id=parent_id,
            mode=mode,
            priority=priority if priority is not None else parent.priority,
            deadline=parent.deadline,
            arrival_ts=self._clock.now(),
        )
        self._units[unit_id] = unit
        parent.children.append(unit_id)
        self._journal.append(
            "unit_created",
            unit_id=unit_id,
            payload={
                "kind": unit.kind.value,
                "label": label,
                "parent_id": parent_id,
                "mode": mode.value,
                "arrival_ts": unit.arrival_ts,
                "deadline": unit.deadline,
            },
        )
        return unit

    def create_tool_call(self, parent_id: UnitId, label: str) -> ExecutionUnit:
        parent = self._units[parent_id]
        index = len(parent.children)
        unit_id = f"{parent_id}/tool-{index}"
        unit = ExecutionUnit(
            unit_id=unit_id,
            run_id=parent.run_id,
            kind=UnitKind.TOOL_CALL,
            label=label,
            parent_id=parent_id,
            priority=parent.priority,
            deadline=parent.deadline,
            arrival_ts=self._clock.now(),
        )
        self._units[unit_id] = unit
        parent.children.append(unit_id)
        self._journal.append(
            "unit_created",
            unit_id=unit_id,
            payload={"kind": unit.kind.value, "label": label, "parent_id": parent_id},
        )
        return unit

    # --- 查询 -------------------------------------------------------------
    def get(self, unit_id: UnitId) -> ExecutionUnit:
        return self._units[unit_id]

    def all_units(self) -> list[ExecutionUnit]:
        return list(self._units.values())

    def descendants(self, unit_id: UnitId) -> list[UnitId]:
        out: list[UnitId] = []
        stack = list(self._units[unit_id].children)
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(self._units[cur].children)
        return out

    def children_all_terminal(self, unit_id: UnitId) -> bool:
        return all(self._units[c].is_terminal for c in self.descendants(unit_id))

    # --- 转移 -------------------------------------------------------------
    def transition(
        self,
        unit_id: UnitId,
        dst: Phase,
        *,
        cause_seq: int | None = None,
        reason: str | None = None,
    ) -> None:
        unit = self._units[unit_id]
        src = unit.phase
        try:
            check_phase_transition(src, dst, kind=unit.kind)
        except IllegalTransition:
            self._journal.append(
                "transition_rejected",
                unit_id=unit_id,
                phase_from=src.value,
                phase_to=dst.value,
                cause_seq=cause_seq,
            )
            raise
        unit.phase = dst
        self._journal.append(
            "phase_transitioned",
            unit_id=unit_id,
            phase_from=src.value,
            phase_to=dst.value,
            cause_seq=cause_seq,
            payload={
                "from": src.value,
                "to": dst.value,
                "ts": self._clock.now(),
                "reason": reason,
            },
        )

    # --- join / cancel ----------------------------------------------------
    def join(self, child_id: UnitId) -> ExecutionUnit:
        child = self._units[child_id]
        if not child.is_terminal:
            raise JoinBeforeTermination(f"{child_id} is {child.phase}")
        parent = self._units[child.parent_id] if child.parent_id else None
        if parent is not None:
            parent.aggregate.recompute_tokens += child.aggregate.recompute_tokens
            parent.aggregate.park_count += child.aggregate.park_count
            parent.aggregate.tasks_completed += child.aggregate.tasks_completed
        self._journal.append(
            "lease_released",
            unit_id=child_id,
            payload={"reason": "join", "recompute_tokens": child.aggregate.recompute_tokens},
        )
        return child

    def register_cancel_hook(self, hook: CancelHook) -> None:
        self._cancel_hooks.append(hook)

    async def cancel(self, unit_id: UnitId, reason: str = "cancelled") -> None:
        """深度优先:先取消后代,再取消自身;每个方向都等待钩子完成。"""
        async with self._lock:
            targets = self.descendants(unit_id) + [unit_id]
            for uid in targets:
                unit = self._units[uid]
                if unit.is_terminal:
                    continue
                for hook in self._cancel_hooks:
                    await hook(uid)
                self.transition(uid, Phase.CANCELLED, reason=reason)

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/unit/test_units.py -q`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat(units): execution unit tree with fork/join and tree-scoped cancel"
```

---

## Task 6: 进程回收器与工具执行

**Files:**
- Create: `src/phaseweaver/tools/__init__.py`, `tools/base.py`, `tools/reaper.py`,
  `tools/python_exec.py`, `tools/shell_exec.py`, `tools/browser_exec.py`, `tools/supervisor.py`
- Test: `tests/unit/test_tools.py`, `tests/integration/test_cancel_subprocess.py`

- [ ] **Step 1: 写 `src/phaseweaver/tools/base.py`**

```python
"""工具抽象:执行器与结果信封。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    KILLED = "killed"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    executor: str  # "python" | "shell" | "browser"
    timeout_s: float = 30.0
    grace_ms: int = 2000
    expected_duration_s: float | None = None
    idempotency_key: str | None = None


@dataclass
class ToolResult:
    status: ToolStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: int = 0
    truncated: bool = False
    started_at: float = 0.0
    finished_at: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK


class ToolExecutor(Protocol):
    name: str

    async def run(self, args: dict[str, Any], spec: ToolSpec) -> ToolResult: ...


def now() -> float:
    return time.monotonic()
```

- [ ] **Step 2: 写 `src/phaseweaver/tools/reaper.py`**

```python
"""跨平台进程树回收。

Windows 没有 SIGTERM 语义:必须用 CREATE_NEW_PROCESS_GROUP + taskkill /T /F。
POSIX 用 start_new_session + killpg。
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys


def make_creation_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
            "start_new_session": False,
        }
    return {"start_new_session": True}


async def terminate_tree(proc: asyncio.subprocess.Process, grace_ms: int) -> None:
    if proc.returncode is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_ms / 1000)
            return
        except TimeoutError:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
    try:
        await asyncio.wait_for(proc.wait(), timeout=max(1.0, grace_ms / 1000))
    except TimeoutError:
        return


async def tree_is_gone(proc: asyncio.subprocess.Process) -> bool:
    if proc.returncode is not None:
        return True
    if sys.platform != "win32":
        try:
            os.killpg(os.getpgid(proc.pid), 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {proc.pid}"], capture_output=True, text=True, check=False
    )
    return str(proc.pid) not in (out.stdout or "")
```

- [ ] **Step 3: 写 `tools/python_exec.py`**

```python
"""Python 代码执行器:隔离子进程 + JSON 行协议。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from typing import Any

from phaseweaver.tools.base import ToolResult, ToolSpec, ToolStatus, now
from phaseweaver.tools.reaper import make_creation_kwargs, terminate_tree

_WRAPPER = "import json,sys,runpy,io\ncode=sys.stdin.read()\ng={'__name__':'__main__'}\nbuf=io.StringIO()\nimport contextlib\nwith contextlib.redirect_stdout(buf):\n    exec(compile(code,'<tool>','exec'),g)\nsys.stdout.write(json.dumps({'ok':True,'stdout':buf.getvalue()}))\n"


class PythonExecutor:
    name = "python"

    async def run(self, args: dict[str, Any], spec: ToolSpec) -> ToolResult:
        code = str(args.get("code", ""))
        started = now()
        with tempfile.TemporaryDirectory(prefix="pw-tool-") as workdir:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-c",
                _WRAPPER,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                **make_creation_kwargs(),  # type: ignore[arg-type]
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(code.encode("utf-8")), timeout=spec.timeout_s
                )
                status = ToolStatus.OK if proc.returncode == 0 else ToolStatus.ERROR
            except TimeoutError:
                await terminate_tree(proc, spec.grace_ms)
                stdout, stderr = b"", b"timed out"
                status = ToolStatus.TIMEOUT
        return ToolResult(
            status=status,
            stdout=stdout.decode("utf-8", "replace")[: 1 << 20],
            stderr=stderr.decode("utf-8", "replace")[: 1 << 20],
            exit_code=proc.returncode,
            duration_ms=int((now() - started) * 1000),
            started_at=started,
            finished_at=now(),
        )
```

- [ ] **Step 4: 写 `tools/shell_exec.py` 与 `tools/browser_exec.py`**

```python
# shell_exec.py
from __future__ import annotations

import asyncio
from typing import Any

from phaseweaver.tools.base import ToolResult, ToolSpec, ToolStatus, now
from phaseweaver.tools.reaper import make_creation_kwargs, terminate_tree


class ShellExecutor:
    name = "shell"

    async def run(self, args: dict[str, Any], spec: ToolSpec) -> ToolResult:
        command = str(args.get("command", ""))
        started = now()
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **make_creation_kwargs(),  # type: ignore[arg-type]
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=spec.timeout_s)
            status = ToolStatus.OK if proc.returncode == 0 else ToolStatus.ERROR
        except TimeoutError:
            await terminate_tree(proc, spec.grace_ms)
            stdout, stderr = b"", b"timed out"
            status = ToolStatus.TIMEOUT
        return ToolResult(
            status=status,
            stdout=stdout.decode("utf-8", "replace")[: 1 << 20],
            stderr=stderr.decode("utf-8", "replace")[: 1 << 20],
            exit_code=proc.returncode,
            duration_ms=int((now() - started) * 1000),
            started_at=started,
            finished_at=now(),
        )
```

```python
# browser_exec.py —— 只定义接口,不引入 Playwright(规格 §2.2 非目标)
from __future__ import annotations

from typing import Any

from phaseweaver.tools.base import ToolResult, ToolSpec, ToolStatus, now


class BrowserExecutor:
    name = "browser"

    async def run(self, args: dict[str, Any], spec: ToolSpec) -> ToolResult:
        ts = now()
        return ToolResult(
            status=ToolStatus.ERROR,
            stderr="browser executor not implemented (phase 1 defines the interface only)",
            started_at=ts,
            finished_at=ts,
        )
```

- [ ] **Step 5: 写 `tools/supervisor.py`**

```python
"""工具监管:按 unit 记录运行中的进程,支持按 unit 取消。"""

from __future__ import annotations

import asyncio
from typing import Any

from phaseweaver.journal import EventJournal
from phaseweaver.tools.base import ToolExecutor, ToolResult, ToolSpec, ToolStatus
from phaseweaver.tools.browser_exec import BrowserExecutor
from phaseweaver.tools.python_exec import PythonExecutor
from phaseweaver.tools.shell_exec import ShellExecutor


class ToolSupervisor:
    def __init__(self, journal: EventJournal) -> None:
        self._journal = journal
        self._executors: dict[str, ToolExecutor] = {
            e.name: e for e in (PythonExecutor(), ShellExecutor(), BrowserExecutor())
        }
        self._live: dict[str, set[asyncio.Task[ToolResult]]] = {}

    async def run(self, unit_id: str, spec: ToolSpec, args: dict[str, Any]) -> ToolResult:
        executor = self._executors[spec.executor]
        self._journal.append(
            "tool_started",
            unit_id=unit_id,
            payload={"tool": spec.name, "executor": spec.executor},
        )
        task: asyncio.Task[ToolResult] = asyncio.create_task(executor.run(args, spec))
        self._live.setdefault(unit_id, set()).add(task)
        try:
            result = await task
        finally:
            self._live[unit_id].discard(task)
        self._journal.append(
            "tool_finished",
            unit_id=unit_id,
            payload={
                "tool": spec.name,
                "status": result.status.value,
                "duration_ms": result.duration_ms,
            },
        )
        return result

    async def cancel_unit(self, unit_id: str) -> None:
        tasks = self._live.get(unit_id, set())
        for task in list(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._journal.append("tool_killed", unit_id=unit_id, payload={"reason": "cancel"})

    async def shutdown(self) -> None:
        for unit_id in list(self._live):
            await self.cancel_unit(unit_id)
```

- [ ] **Step 6: 写测试 `tests/integration/test_cancel_subprocess.py`**

```python
import asyncio
import sys

from phaseweaver.tools.base import ToolSpec, ToolStatus
from phaseweaver.tools.python_exec import PythonExecutor
from phaseweaver.tools.shell_exec import ShellExecutor


def test_python_exec_returns_stdout() -> None:
    async def scenario() -> None:
        result = await PythonExecutor().run({"code": "print('hi')"}, ToolSpec("p", "python"))
        assert result.status is ToolStatus.OK
        assert "hi" in result.stdout

    asyncio.run(scenario())


def test_timeout_kills_process() -> None:
    async def scenario() -> None:
        spec = ToolSpec("s", "shell", timeout_s=0.5, grace_ms=500)
        cmd = "python -c \"import time; time.sleep(30)\""
        result = await ShellExecutor().run({"command": cmd}, spec)
        assert result.status is ToolStatus.TIMEOUT
        assert result.exit_code is not None

    asyncio.run(scenario())


def test_python_error_reported() -> None:
    async def scenario() -> None:
        result = await PythonExecutor().run({"code": "raise ValueError('boom')"}, ToolSpec("p", "python"))
        assert result.status is ToolStatus.ERROR
        assert "ValueError" in result.stderr

    asyncio.run(scenario())
```

- [ ] **Step 7: 运行测试**

Run: `uv run pytest tests/unit/test_tools.py tests/integration/test_cancel_subprocess.py -q`
Expected: 全绿。若 Windows 上 `test_timeout_kills_process` 卡住超过 10 秒,说明 `taskkill` 未生效,检查 `make_creation_kwargs()` 是否被 `**` 正确展开。

- [ ] **Step 8: 提交**

```bash
git add -A && git commit -m "feat(tools): cross-platform process reaper with python/shell executors"
```

---

## Task 7: 租约、预算与不变量 I1–I4

**Files:**
- Create: `src/phaseweaver/leases.py`, `src/phaseweaver/invariant.py`
- Test: `tests/unit/test_leases.py`, `tests/property/test_invariants.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_leases.py`**

```python
from phaseweaver.clock import VirtualClock
from phaseweaver.journal import EventJournal
from phaseweaver.leases import Budget, LeaseBroker
from phaseweaver.types import LeaseKind


def make_broker(total: int = 1000) -> LeaseBroker:
    clock = VirtualClock()
    return LeaseBroker(
        journal=EventJournal(run_id="r1", clock=clock),
        clock=clock,
        budget=Budget(total_bytes=total, headroom_frac=0.0),
    )


def test_acquire_within_budget() -> None:
    broker = make_broker()
    lease = broker.acquire("u1", "c1", LeaseKind.KV_BYTES, 600, priority=5)
    assert lease is not None
    assert broker.used(LeaseKind.KV_BYTES) == 600


def test_acquire_over_budget_returns_none() -> None:
    broker = make_broker()
    assert broker.acquire("u1", "c1", LeaseKind.KV_BYTES, 600, priority=5) is not None
    assert broker.acquire("u2", "c2", LeaseKind.KV_BYTES, 600, priority=5) is None
    assert broker.used(LeaseKind.KV_BYTES) == 600


def test_release_frees_capacity() -> None:
    broker = make_broker()
    lease = broker.acquire("u1", "c1", LeaseKind.KV_BYTES, 1000, priority=5)
    assert lease is not None
    broker.release(lease.lease_id)
    assert broker.used(LeaseKind.KV_BYTES) == 0
    assert broker.acquire("u2", "c2", LeaseKind.KV_BYTES, 1000, priority=5) is not None


def test_headroom_is_reserved() -> None:
    broker = LeaseBroker(
        journal=EventJournal(run_id="r1", clock=VirtualClock()),
        clock=VirtualClock(),
        budget=Budget(total_bytes=1000, headroom_frac=0.1),
    )
    assert broker.available(LeaseKind.KV_BYTES) == 900
```

- [ ] **Step 2: 写失败测试 `tests/property/test_invariants.py`**

```python
"""I1–I4 的 property test:随机状态组合,不变量永不违反。"""

from hypothesis import given, settings, strategies as st

from phaseweaver.clock import VirtualClock
from phaseweaver.invariant import InvariantChecker
from phaseweaver.journal import EventJournal
from phaseweaver.leases import Budget, LeaseBroker
from phaseweaver.phases import NON_GPU_PHASES
from phaseweaver.types import ContextState, InferenceContext, LeaseKind, Phase
from phaseweaver.units import UnitRegistry


@settings(max_examples=50, deadline=None)
@given(
    phase=st.sampled_from(list(Phase)),
    state=st.sampled_from(list(ContextState)),
    amount=st.integers(min_value=0, max_value=1000),
)
def test_i1_no_active_context_in_non_gpu_phase(
    phase: Phase, state: ContextState, amount: int
) -> None:
    clock = VirtualClock()
    journal = EventJournal(run_id="r1", clock=clock)
    reg = UnitRegistry(journal=journal, clock=clock)
    broker = LeaseBroker(journal=journal, clock=clock, budget=Budget(2000, 0.0))
    unit = reg.create_root("x")
    unit.phase = phase
    unit.context = InferenceContext(
        context_id="c1", state=state, resident_bytes=amount
    )
    if state in (ContextState.ACTIVE, ContextState.RETAINED) and amount > 0:
        broker.acquire(unit.unit_id, "c1", LeaseKind.KV_BYTES, amount, 5)
    checker = InvariantChecker(registry=reg, broker=broker, journal=journal, strict=False)
    violations = checker.check()

    i1 = [v for v in violations if v.code == "I1"]
    if phase in NON_GPU_PHASES and state is ContextState.ACTIVE:
        assert i1, "I1 must fire for ACTIVE context in a non-GPU phase"
    else:
        assert not i1
```

- [ ] **Step 3: 运行确认失败**

Run: `uv run pytest tests/unit/test_leases.py tests/property/test_invariants.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.leases`

- [ ] **Step 4: 写 `src/phaseweaver/leases.py`**

```python
"""租约 = 处置权 + 成本(ADR-0002)。只做决策与记账,不触碰引擎。"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from phaseweaver.clock import Clock
from phaseweaver.journal import EventJournal
from phaseweaver.types import LeaseId, LeaseKind


@dataclass(frozen=True)
class Budget:
    total_bytes: int
    headroom_frac: float = 0.10
    max_per_agent_factor: float = 1.5

    def capacity(self) -> int:
        return int(self.total_bytes * (1.0 - self.headroom_frac))


@dataclass
class Lease:
    lease_id: LeaseId
    unit_id: str
    context_id: str | None
    kind: LeaseKind
    amount: int
    priority: int
    expires_at: float | None = None
    active: bool = True


class LeaseBroker:
    def __init__(self, journal: EventJournal, clock: Clock, budget: Budget) -> None:
        self._journal = journal
        self._clock = clock
        self._budget = budget
        self._leases: dict[LeaseId, Lease] = {}
        self._seq = itertools.count(1)

    @property
    def budget(self) -> Budget:
        return self._budget

    def _used(self, kind: LeaseKind) -> int:
        return sum(l.amount for l in self._leases.values() if l.active and l.kind is kind)

    def used(self, kind: LeaseKind) -> int:
        return self._used(kind)

    def available(self, kind: LeaseKind) -> int:
        if kind is LeaseKind.KV_BYTES:
            return self._budget.capacity() - self._used(kind)
        return 1_000_000

    def acquire(
        self,
        unit_id: str,
        context_id: str | None,
        kind: LeaseKind,
        amount: int,
        priority: int,
        expires_at: float | None = None,
    ) -> Lease | None:
        self._journal.append(
            "lease_requested",
            unit_id=unit_id,
            payload={"kind": kind.value, "amount": amount, "priority": priority},
        )
        if amount > self.available(kind):
            self._journal.append(
                "admission_rejected",
                unit_id=unit_id,
                payload={"kind": kind.value, "amount": amount, "reason": "budget"},
            )
            return None
        lease = Lease(
            lease_id=f"L{next(self._seq)}",
            unit_id=unit_id,
            context_id=context_id,
            kind=kind,
            amount=amount,
            priority=priority,
            expires_at=expires_at,
        )
        self._leases[lease.lease_id] = lease
        self._journal.append(
            "lease_granted",
            unit_id=unit_id,
            payload={
                "lease_id": lease.lease_id,
                "kind": kind.value,
                "amount": amount,
                "context_id": context_id,
            },
        )
        return lease

    def release(self, lease_id: LeaseId, reason: str = "released") -> None:
        lease = self._leases.get(lease_id)
        if lease is None or not lease.active:
            return
        lease.active = False
        self._journal.append(
            "lease_released",
            unit_id=lease.unit_id,
            payload={"lease_id": lease_id, "kind": lease.kind.value, "reason": reason},
        )

    def active_leases(self, kind: LeaseKind | None = None) -> list[Lease]:
        return [
            l
            for l in self._leases.values()
            if l.active and (kind is None or l.kind is kind)
        ]

    def lease_for_context(self, context_id: str, kind: LeaseKind) -> Lease | None:
        for lease in self._leases.values():
            if lease.active and lease.kind is kind and lease.context_id == context_id:
                return lease
        return None

    def release_all(self, reason: str = "shutdown") -> None:
        for lease in self.active_leases():
            self.release(lease.lease_id, reason=reason)
        self._journal.append("forced_release", payload={"reason": reason})
```

- [ ] **Step 5: 写 `src/phaseweaver/invariant.py`**

```python
"""不变量校验。strict 模式抛错并 dump 上下文,audit 模式只记事件。"""

from __future__ import annotations

from dataclasses import dataclass

from phaseweaver.errors import PhaseWeaverError
from phaseweaver.journal import EventJournal
from phaseweaver.leases import LeaseBroker
from phaseweaver.phases import NON_GPU_PHASES
from phaseweaver.types import ContextState, LeaseKind, Phase
from phaseweaver.units import UnitRegistry


@dataclass(frozen=True)
class Violation:
    code: str
    unit_id: str | None
    detail: str


class InvariantError(PhaseWeaverError):
    def __init__(self, violations: list[Violation]) -> None:
        super().__init__("; ".join(f"{v.code}:{v.unit_id}:{v.detail}" for v in violations))
        self.violations = violations


class InvariantChecker:
    def __init__(
        self,
        registry: UnitRegistry,
        broker: LeaseBroker,
        journal: EventJournal,
        *,
        strict: bool = True,
    ) -> None:
        self._registry = registry
        self._broker = broker
        self._journal = journal
        self._strict = strict

    def check(self) -> list[Violation]:
        violations: list[Violation] = []
        violations.extend(self._check_i1())
        violations.extend(self._check_i2())
        violations.extend(self._check_i3())
        violations.extend(self._check_i4())
        for v in violations:
            self._journal.append(
                "invariant_violated",
                unit_id=v.unit_id,
                payload={"code": v.code, "detail": v.detail},
            )
        if violations and self._strict:
            raise InvariantError(violations)
        return violations

    def _check_i1(self) -> list[Violation]:
        out: list[Violation] = []
        for unit in self._registry.all_units():
            ctx = unit.context
            if ctx is None:
                continue
            if unit.phase in NON_GPU_PHASES and ctx.state is ContextState.ACTIVE:
                out.append(
                    Violation("I1", unit.unit_id, f"{unit.phase} holds ACTIVE context")
                )
        return out

    def _check_i2(self) -> list[Violation]:
        out: list[Violation] = []
        for unit in self._registry.all_units():
            ctx = unit.context
            if ctx is None:
                continue
            lease = self._broker.lease_for_context(ctx.context_id, LeaseKind.KV_BYTES)
            occupied = ctx.state in (ContextState.ACTIVE, ContextState.RETAINED)
            if occupied and lease is None:
                out.append(Violation("I2", unit.unit_id, "occupies GPU without lease"))
            if not occupied and lease is not None:
                out.append(Violation("I2", unit.unit_id, "lease without GPU residency"))
        return out

    def _check_i3(self) -> list[Violation]:
        used = self._broker.used(LeaseKind.KV_BYTES)
        cap = self._broker.budget.capacity()
        if used > cap:
            return [Violation("I3", None, f"kv used {used} > capacity {cap}")]
        return []

    def _check_i4(self) -> list[Violation]:
        out: list[Violation] = []
        for unit in self._registry.all_units():
            if not unit.phase.is_terminal:
                continue
            held = [l for l in self._broker.active_leases() if l.unit_id == unit.unit_id]
            if held:
                out.append(Violation("I4", unit.unit_id, f"{len(held)} leases after terminal"))
            if unit.context is not None and unit.context.state is not ContextState.RELEASED:
                out.append(
                    Violation("I4", unit.unit_id, f"context {unit.context.state} after terminal")
                )
        return out
```

- [ ] **Step 6: 运行测试**

Run: `uv run pytest tests/unit/test_leases.py tests/property/test_invariants.py -q`
Expected: 全绿(I1 的 property 断言成立)。

> 若 mypy 报 `_used` 与 `used` 重复,保留两者(`_used` 供内部、`used` 供外部),在 `_used` 上加 `# noqa: D401` 不需要。若报 `Budget(2000, 0.0)` 位置参数问题,改成关键字形式。

- [ ] **Step 7: 提交**

```bash
git add -A && git commit -m "feat(leases): budget, lease broker and invariant checker (I1-I4)"
```

---

## Task 8: 调度器协议与局部策略(B0/B1/B2a)

**Files:**
- Create: `src/phaseweaver/policy.py`
- Test: `tests/unit/test_policy.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_policy.py`**

```python
from phaseweaver.policy import (
    ClusterSnapshot,
    PinAlwaysPolicy,
    TtlEvictPolicy,
    YieldOnToolPolicy,
)
from phaseweaver.types import Disposition


def snap(resident: int = 0, capacity: int = 8 * 1024**3) -> ClusterSnapshot:
    return ClusterSnapshot(
        now=0.0,
        capacity_bytes=capacity,
        resident_bytes=resident,
        queued=0,
        reasoning=1,
        slack_by_unit={"u1": 60.0},
        recent_thrash={},
    )


def test_pin_always_never_yields() -> None:
    assert PinAlwaysPolicy().disposition_for("u1", snap(), 60.0) is Disposition.RETAIN


def test_ttl_evict_uses_ttl() -> None:
    policy = TtlEvictPolicy(ttl_s=5.0)
    assert policy.disposition_for("u1", snap(), 2.0) is Disposition.RETAIN
    assert policy.disposition_for("u1", snap(), 10.0) is Disposition.DROP


def test_yield_buckets() -> None:
    policy = YieldOnToolPolicy(retain_max_s=2.0, offload_max_s=20.0)
    assert policy.disposition_for("u1", snap(), 1.0) is Disposition.RETAIN
    assert policy.disposition_for("u1", snap(), 10.0) is Disposition.OFFLOAD
    assert policy.disposition_for("u1", snap(), 30.0) is Disposition.DROP


def test_yield_degrades_when_budget_tight() -> None:
    policy = YieldOnToolPolicy(retain_max_s=2.0, offload_max_s=20.0)
    tight = snap(resident=int(0.95 * 8 * 1024**3))
    assert policy.disposition_for("u1", tight, 1.0) is Disposition.OFFLOAD
    assert policy.disposition_for("u1", tight, 10.0) is Disposition.DROP


def test_yield_without_offload_capability() -> None:
    policy = YieldOnToolPolicy(retain_max_s=2.0, offload_max_s=20.0, offload_available=False)
    assert policy.disposition_for("u1", snap(), 10.0) is Disposition.DROP
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/unit/test_policy.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.policy`

- [ ] **Step 3: 写 `src/phaseweaver/policy.py`**

```python
"""调度策略:只看只读快照,不直接改状态(ADR-0005)。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from phaseweaver.types import Disposition


@dataclass(frozen=True)
class ClusterSnapshot:
    now: float
    capacity_bytes: int
    resident_bytes: int
    queued: int
    reasoning: int
    slack_by_unit: dict[str, float]
    recent_thrash: dict[str, int]

    @property
    def pressure(self) -> float:
        if self.capacity_bytes == 0:
            return 1.0
        return self.resident_bytes / self.capacity_bytes


class Scheduler(Protocol):
    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool: ...
    def disposition_for(self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float) -> Disposition: ...
    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool: ...


class PinAlwaysPolicy:
    """B0:一律保留,无让出、无抢占。"""

    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return True

    def disposition_for(
        self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float
    ) -> Disposition:
        return Disposition.RETAIN

    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return True


class TtlEvictPolicy:
    """B1:工具时长超过 TTL 即释放(Continuum 式)。"""

    def __init__(self, ttl_s: float = 5.0) -> None:
        self.ttl_s = ttl_s

    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return True

    def disposition_for(
        self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float
    ) -> Disposition:
        return Disposition.RETAIN if expected_tool_s < self.ttl_s else Disposition.DROP

    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return True


class YieldOnToolPolicy:
    """B2a:逐单元分桶的局部规则(仅用于打通机制与消融)。"""

    def __init__(
        self,
        retain_max_s: float = 2.0,
        offload_max_s: float = 20.0,
        *,
        offload_available: bool = True,
        pressure_threshold: float = 0.8,
    ) -> None:
        self.retain_max_s = retain_max_s
        self.offload_max_s = offload_max_s
        self.offload_available = offload_available
        self.pressure_threshold = pressure_threshold

    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return True

    def _base(
        self, expected_tool_s: float, *, offload_available: bool
    ) -> Disposition:
        if expected_tool_s < self.retain_max_s:
            return Disposition.RETAIN
        if offload_available and expected_tool_s < self.offload_max_s:
            return Disposition.OFFLOAD
        return Disposition.DROP

    def disposition_for(
        self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float
    ) -> Disposition:
        base = self._base(expected_tool_s, offload_available=self.offload_available)
        if snap.pressure <= self.pressure_threshold:
            return base
        ladder = [Disposition.RETAIN, Disposition.OFFLOAD, Disposition.DROP]
        degraded = ladder[min(ladder.index(base) + 1, len(ladder) - 1)]
        if degraded is Disposition.OFFLOAD and not self.offload_available:
            return Disposition.DROP
        return degraded

    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return snap.pressure < 0.98
```

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/unit/test_policy.py -q`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat(policy): scheduler protocol and baselines B0/B1/B2a"
```

---

## Task 9: Mock 后端(块分配 + 引擎事件)

**Files:**
- Create: `src/phaseweaver/backends/__init__.py`, `backends/base.py`, `backends/mock.py`
- Test: `tests/unit/test_mock_backend.py`

- [ ] **Step 1: 写 `src/phaseweaver/backends/base.py`**

```python
"""推理后端协议:PhaseWeaver 与引擎之间唯一的契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from phaseweaver.types import ContextId, Disposition


@dataclass(frozen=True)
class BackendCaps:
    capacity_bytes: int
    block_tokens: int
    bytes_per_token: int
    prefill_tok_per_s: int
    decode_tok_per_s: int
    mem_bw_bps: int
    supports_retain: bool = True
    supports_drop: bool = True
    supports_offload: bool = False
    supports_engine_sleep: bool = False

    @property
    def blocks_for(self) -> int:
        return max(1, self.capacity_bytes // (self.block_tokens * self.bytes_per_token))


class EngineEventKind(StrEnum):
    BLOCK_STORED = "block_stored"
    BLOCK_REMOVED = "block_removed"
    REQUEST_PREEMPTED = "request_preempted"
    QUEUE_DEPTH = "queue_depth"


@dataclass(frozen=True)
class EngineEvent:
    kind: EngineEventKind
    context_id: ContextId | None
    blocks: int = 0
    payload: dict[str, object] | None = None


@dataclass
class Handle:
    unit_id: str
    context_id: ContextId
    resident_tokens: int = 0
    dropped: bool = False


@dataclass(frozen=True)
class GenStats:
    prompt_tokens: int
    completion_tokens: int
    start_ts: float
    end_ts: float
    ttft_ms: float
    recompute_tokens: int = 0


@dataclass(frozen=True)
class ParkResult:
    applied: Disposition
    degraded_from: Disposition | None
    moved_bytes: int
    latency_ms: float


@dataclass(frozen=True)
class RestoreResult:
    recompute_tokens: int
    loaded_bytes: int
    latency_ms: float


class InferenceBackend(Protocol):
    def caps(self) -> BackendCaps: ...
    def available_bytes(self) -> int: ...
    async def submit(self, unit_id: str, context_id: str, tokens: int) -> Handle: ...
    async def complete(self, handle: Handle, out_tokens: int) -> GenStats: ...
    async def abort(self, handle: Handle, reason: str) -> None: ...
    async def park(self, handle: Handle, disposition: Disposition) -> ParkResult: ...
    async def restore(self, handle: Handle) -> RestoreResult: ...
```

- [ ] **Step 2: 写失败测试 `tests/unit/test_mock_backend.py`**

```python
import asyncio

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.journal import EventJournal
from phaseweaver.types import Disposition


def caps() -> BackendCaps:
    return BackendCaps(
        capacity_bytes=1024 * 1024,
        block_tokens=16,
        bytes_per_token=1000,
        prefill_tok_per_s=1000,
        decode_tok_per_s=100,
        mem_bw_bps=10**7,
        supports_offload=True,
    )


def make() -> MockBackend:
    clock = VirtualClock()
    clock.start_driver()
    journal = EventJournal(run_id="r1", clock=clock)
    return MockBackend(caps=caps(), clock=clock, journal=journal)


def test_submit_allocates_blocks() -> None:
    async def scenario() -> None:
        backend = make()
        handle = await backend.submit("u1", "c1", 32)
        assert handle.resident_tokens == 32
        assert backend.available_bytes() == 1024 * 1024 - 32 * 1000

    asyncio.run(scenario())


def test_park_drop_frees_and_restore_recomputes() -> None:
    async def scenario() -> None:
        backend = make()
        handle = await backend.submit("u1", "c1", 64)
        park = await backend.park(handle, Disposition.DROP)
        assert park.applied is Disposition.DROP
        assert backend.available_bytes() == 1024 * 1024
        restore = await backend.restore(handle)
        assert restore.recompute_tokens == 64
        assert backend.available_bytes() == 1024 * 1024 - 64 * 1000

    asyncio.run(scenario())


def test_park_retain_keeps_residency() -> None:
    async def scenario() -> None:
        backend = make()
        handle = await backend.submit("u1", "c1", 64)
        before = backend.available_bytes()
        park = await backend.park(handle, Disposition.RETAIN)
        assert park.applied is Disposition.RETAIN
        assert backend.available_bytes() == before

    asyncio.run(scenario())


def test_offload_moves_bytes_with_cost() -> None:
    async def scenario() -> None:
        backend = make()
        handle = await backend.submit("u1", "c1", 64)
        park = await backend.park(handle, Disposition.OFFLOAD)
        assert park.applied is Disposition.OFFLOAD
        assert park.moved_bytes == 64 * 1000
        assert backend.available_bytes() == 1024 * 1024

    asyncio.run(scenario())
```

- [ ] **Step 3: 运行确认失败**

Run: `uv run pytest tests/unit/test_mock_backend.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.backends`

- [ ] **Step 4: 写 `src/phaseweaver/backends/mock.py`**

```python
"""确定性 mock 推理后端。

- 时间只来自 VirtualClock,不碰真实时间
- KV 以 block 为粒度分配,超限即报 EngineOutOfMemory(说明租约记账有 bug)
- 发出与 vLLM kv_events 同词汇的引擎事件,runtime 走同一条感知路径
"""

from __future__ import annotations

from dataclasses import dataclass

from phaseweaver.backends.base import (
    BackendCaps,
    EngineEvent,
    EngineEventKind,
    GenStats,
    Handle,
    ParkResult,
    RestoreResult,
)
from phaseweaver.clock import Clock
from phaseweaver.errors import EngineOutOfMemory
from phaseweaver.journal import EventJournal
from phaseweaver.types import Disposition


@dataclass
class _Allocation:
    tokens: int
    blocks: int
    offloaded: bool = False


class MockBackend:
    def __init__(
        self,
        caps: BackendCaps,
        clock: Clock,
        journal: EventJournal | None = None,
        *,
        run_id: str = "r1",
    ) -> None:
        self._caps = caps
        self._clock = clock
        self._journal = journal
        self._run_id = run_id
        self._allocs: dict[str, _Allocation] = {}
        self._events: list[EngineEvent] = []
        self._used_blocks = 0

    # --- 协议实现 ---------------------------------------------------------
    def caps(self) -> BackendCaps:
        return self._caps

    def available_bytes(self) -> int:
        return self._caps.capacity_bytes - self._used_blocks * self._block_bytes()

    def engine_events(self) -> list[EngineEvent]:
        return list(self._events)

    async def submit(self, unit_id: str, context_id: str, tokens: int) -> Handle:
        handle = Handle(unit_id=unit_id, context_id=context_id, resident_tokens=tokens)
        self._allocate(context_id, tokens)
        return handle

    async def complete(self, handle: Handle, out_tokens: int) -> GenStats:
        start = self._clock.now()
        recompute = 0
        if handle.dropped:
            recompute = handle.resident_tokens
            self._allocate(handle.context_id, handle.resident_tokens)
            handle.dropped = False
        prompt_tokens = handle.resident_tokens + recompute
        prefill = (out_tokens * 0 + prompt_tokens) / self._caps.prefill_tok_per_s
        decode = out_tokens / self._caps.decode_tok_per_s
        await self._clock.wait(prefill + decode)
        end = self._clock.now()
        return GenStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=out_tokens,
            start_ts=start,
            end_ts=end,
            ttft_ms=prefill * 1000,
            recompute_tokens=recompute,
        )

    async def abort(self, handle: Handle, reason: str) -> None:
        self._free(handle.context_id)
        handle.dropped = True
        self._emit(EngineEventKind.BLOCK_REMOVED, handle.context_id, 0)

    async def park(self, handle: Handle, disposition: Disposition) -> ParkResult:
        start = self._clock.now()
        alloc = self._allocs.get(handle.context_id)
        if alloc is None:
            return ParkResult(disposition, None, 0, 0.0)
        applied = disposition
        if applied is Disposition.OFFLOAD and not self._caps.supports_offload:
            applied = Disposition.DROP
        if applied is Disposition.RETAIN:
            return ParkResult(Disposition.RETAIN, None, 0, 0.0)
        if applied is Disposition.RELEASE:
            self._free(handle.context_id)
            return ParkResult(Disposition.RELEASE, None, 0, 0.0)
        moved = alloc.tokens * self._caps.bytes_per_token
        self._free(handle.context_id)
        handle.dropped = True
        alloc.offloaded = applied is Disposition.OFFLOAD
        self._emit(
            EngineEventKind.REQUEST_PREEMPTED,
            handle.context_id,
            0,
            {"disposition": applied.value},
        )
        cost_s = (moved / self._caps.mem_bw_bps) if applied is Disposition.OFFLOAD else 0.0
        if cost_s > 0:
            await self._clock.wait(cost_s)
        return ParkResult(
            applied=applied,
            degraded_from=disposition if applied is not disposition else None,
            moved_bytes=moved if applied is Disposition.OFFLOAD else 0,
            latency_ms=(self._clock.now() - start) * 1000,
        )

    async def restore(self, handle: Handle) -> RestoreResult:
        start = self._clock.now()
        recompute = 0
        loaded = 0
        alloc = self._allocs.get(handle.context_id)
        if alloc is None or handle.dropped:
            recompute = handle.resident_tokens
            self._allocate(handle.context_id, recompute)
            cost = recompute / self._caps.prefill_tok_per_s
        else:
            cost = 0.0
        if cost > 0:
            await self._clock.wait(cost)
        handle.dropped = False
        return RestoreResult(
            recompute_tokens=recompute,
            loaded_bytes=loaded,
            latency_ms=(self._clock.now() - start) * 1000,
        )

    # --- 内部 -------------------------------------------------------------
    def _block_bytes(self) -> int:
        return self._caps.block_tokens * self._caps.bytes_per_token

    def _allocate(self, context_id: str, tokens: int) -> None:
        blocks = (tokens + self._caps.block_tokens - 1) // self._caps.block_tokens
        if self._used_blocks + blocks > self._caps.blocks_for:
            raise EngineOutOfMemory(
                f"blocks exhausted: used={self._used_blocks} want={blocks} "
                f"cap={self._caps.blocks_for}"
            )
        self._allocs[context_id] = _Allocation(tokens=tokens, blocks=blocks)
        self._used_blocks += blocks
        self._emit(EngineEventKind.BLOCK_STORED, context_id, blocks)

    def _free(self, context_id: str) -> None:
        alloc = self._allocs.pop(context_id, None)
        if alloc is not None:
            self._used_blocks -= alloc.blocks
            self._emit(EngineEventKind.BLOCK_REMOVED, context_id, alloc.blocks)

    def _emit(
        self,
        kind: EngineEventKind,
        context_id: str | None,
        blocks: int,
        payload: dict[str, object] | None = None,
    ) -> None:
        self._events.append(
            EngineEvent(kind=kind, context_id=context_id, blocks=blocks, payload=payload)
        )
```

- [ ] **Step 5: 运行测试**

Run: `uv run pytest tests/unit/test_mock_backend.py -q`
Expected: `4 passed`

- [ ] **Step 6: 提交**

```bash
git add -A && git commit -m "feat(backends): mock inference backend with block accounting"
```

---

## Task 10: Runtime 与 Agent API(接线)

**Files:**
- Create: `src/phaseweaver/runtime.py`
- Test: `tests/integration/test_runtime_flow.py`

- [ ] **Step 1: 写失败测试 `tests/integration/test_runtime_flow.py`**

```python
import asyncio

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.mock import MockBackend
from phaseweaver.clock import VirtualClock
from phaseweaver.leases import Budget
from phaseweaver.policy import PinAlwaysPolicy, YieldOnToolPolicy
from phaseweaver.runtime import Runtime, RuntimeConfig
from phaseweaver.tools.base import ToolSpec


def caps() -> BackendCaps:
    return BackendCaps(
        capacity_bytes=4 * 1024 * 1024,
        block_tokens=16,
        bytes_per_token=1000,
        prefill_tok_per_s=1000,
        decode_tok_per_s=100,
        mem_bw_bps=10**7,
        supports_offload=True,
    )


async def agent(unit) -> None:  # type: ignore[no-untyped-def]
    await unit.reason(prompt_tokens=256, out_tokens=32)
    await unit.call_tool(ToolSpec("sleepy", "python", expected_duration_s=30.0), {"code": "print(1)"})
    await unit.reason(prompt_tokens=320, out_tokens=32)


def test_runtime_completes_and_releases() -> None:
    async def scenario() -> None:
        clock = VirtualClock()
        clock.start_driver()
        rt = Runtime(
            backend=MockBackend(caps=caps(), clock=clock, journal=None),  # type: ignore[arg-type]
            clock=clock,
            budget=Budget(total_bytes=caps().capacity_bytes, headroom_frac=0.0),
            policy=PinAlwaysPolicy(),
            config=RuntimeConfig(run_id="t1"),
        )
        await rt.start()
        outcome = await rt.run_task("demo", agent, slo_s=600.0)
        await rt.shutdown()
        assert outcome.phase == "COMPLETED"
        assert rt.journal.by_type("phase_transitioned")

    asyncio.run(scenario())


def test_yield_policy_parks_before_tool() -> None:
    async def scenario() -> None:
        clock = VirtualClock()
        clock.start_driver()
        rt = Runtime(
            backend=MockBackend(caps=caps(), clock=clock, journal=None),  # type: ignore[arg-type]
            clock=clock,
            budget=Budget(total_bytes=caps().capacity_bytes, headroom_frac=0.0),
            policy=YieldOnToolPolicy(retain_max_s=1.0, offload_max_s=60.0),
            config=RuntimeConfig(run_id="t2"),
        )
        await rt.start()
        await rt.run_task("demo", agent, slo_s=600.0)
        await rt.shutdown()
        assert rt.journal.by_type("park_completed")

    asyncio.run(scenario())
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/integration/test_runtime_flow.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.runtime`

- [ ] **Step 3: 写 `src/phaseweaver/runtime.py`**

```python
"""Runtime:把阶段、租约、处置、工具、日志接线到一起。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phaseweaver.clock import Clock
from phaseweaver.errors import AdmissionRejected
from phaseweaver.invariant import InvariantChecker
from phaseweaver.journal import EventJournal
from phaseweaver.leases import Budget, Lease, LeaseBroker
from phaseweaver.phases import NON_GPU_PHASES
from phaseweaver.policy import ClusterSnapshot, Scheduler
from phaseweaver.tools.base import ToolResult, ToolSpec
from phaseweaver.tools.supervisor import ToolSupervisor
from phaseweaver.types import (
    ContextState,
    Disposition,
    Phase,
    UnitId,
)
from phaseweaver.units import ExecutionUnit, UnitRegistry

AgentFn = Callable[["AgentHandle"], Awaitable[None]]


@dataclass
class RuntimeConfig:
    run_id: str = "run"
    strict_invariants: bool = True
    journal_path: Path | None = None
    default_expected_tool_s: float = 5.0


@dataclass
class TaskOutcome:
    unit_id: UnitId
    phase: str
    latency_ms: float
    recompute_tokens: int = 0
    error: str | None = None


class AgentHandle:
    """交给用户 Agent 代码的唯一入口。"""

    def __init__(self, runtime: "Runtime", unit: ExecutionUnit) -> None:
        self._rt = runtime
        self.unit = unit

    async def reason(self, *, prompt_tokens: int, out_tokens: int) -> Any:
        return await self._rt._reason(self.unit, prompt_tokens, out_tokens)

    async def call_tool(self, spec: ToolSpec, args: dict[str, Any]) -> ToolResult:
        return await self._rt._call_tool(self.unit, spec, args)

    async def fork(self, label: str, fn: AgentFn) -> TaskOutcome:
        return await self._rt._fork(self.unit, label, fn)


class Runtime:
    def __init__(
        self,
        backend: Any,
        clock: Clock,
        budget: Budget,
        policy: Scheduler,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.clock = clock
        self.backend = backend
        self.journal = EventJournal(
            run_id=self.config.run_id, clock=clock, path=self.config.journal_path
        )
        self.registry = UnitRegistry(journal=self.journal, clock=clock)
        self.broker = LeaseBroker(journal=self.journal, clock=clock, budget=budget)
        self.policy = policy
        self.supervisor = ToolSupervisor(self.journal)
        self.checker = InvariantChecker(
            registry=self.registry,
            broker=self.broker,
            journal=self.journal,
            strict=self.config.strict_invariants,
        )
        self._contexts: dict[UnitId, str] = {}
        self._handles: dict[UnitId, Any] = {}
        self._leases: dict[UnitId, Lease] = {}
        self._park_times: dict[UnitId, list[float]] = {}
        self._started = False

    # --- 生命周期 ---------------------------------------------------------
    async def start(self) -> None:
        if self._started:
            return
        self.registry.register_cancel_hook(self.supervisor.cancel_unit)
        self.journal.append("run_started", payload={"run_id": self.config.run_id})
        self._started = True

    async def shutdown(self) -> None:
        await self.supervisor.shutdown()
        self.broker.release_all()
        for unit in self.registry.all_units():
            if unit.context is not None and unit.context.state is not ContextState.RELEASED:
                unit.context.state = ContextState.RELEASED
        self.journal.append("run_finished")
        self.journal.close()
        self._started = False

    # --- 任务入口 ---------------------------------------------------------
    async def run_task(
        self, label: str, fn: AgentFn, *, slo_s: float = 120.0, priority: int = 5
    ) -> TaskOutcome:
        unit = self.registry.create_root(
            label, priority=priority, deadline=self.clock.now() + slo_s
        )
        started = self.clock.now()
        self.registry.transition(unit.unit_id, Phase.QUEUED)
        try:
            await fn(AgentHandle(self, unit))
            if not unit.phase.is_terminal:
                self.registry.transition(unit.unit_id, Phase.COMPLETED)
            err = None
        except asyncio.CancelledError:
            await self._cancel(unit, "cancelled")
            err = "cancelled"
        except Exception as exc:  # noqa: BLE001
            if not unit.phase.is_terminal:
                self.registry.transition(unit.unit_id, Phase.FAILED)
            err = f"{type(exc).__name__}: {exc}"
        finally:
            await self._release_unit(unit)
        self.checker.check()
        return TaskOutcome(
            unit_id=unit.unit_id,
            phase=unit.phase.value,
            latency_ms=(self.clock.now() - started) * 1000,
            recompute_tokens=unit.aggregate.recompute_tokens,
            error=err,
        )

    # --- 内部:推理 -------------------------------------------------------
    def _snapshot(self) -> ClusterSnapshot:
        units = self.registry.all_units()
        now = self.clock.now()
        slack = {
            u.unit_id: (u.deadline - now) if u.deadline is not None else 1e9 for u in units
        }
        return ClusterSnapshot(
            now=now,
            capacity_bytes=self.backend.caps().capacity_bytes,
            resident_bytes=self.broker.used(self._kv_kind()),
            queued=sum(1 for u in units if u.phase is Phase.QUEUED),
            reasoning=sum(1 for u in units if u.phase is Phase.REASONING),
            slack_by_unit=slack,
            recent_thrash={},
        )

    def _kv_kind(self) -> Any:
        from phaseweaver.types import LeaseKind

        return LeaseKind.KV_BYTES

    async def _reason(self, unit: ExecutionUnit, prompt_tokens: int, out_tokens: int) -> Any:
        if unit.phase is Phase.QUEUED:
            self.registry.transition(unit.unit_id, Phase.REASONING)
        ctx = unit.context
        if ctx is None:
            ctx = self._new_context(unit)
        if ctx.state is not ContextState.ACTIVE:
            await self._activate(unit, ctx, prompt_tokens)
        handle = self._handles[unit.unit_id]
        stats = await self.backend.complete(handle, out_tokens)
        self._set_context(unit, ctx, ContextState.ACTIVE, prompt_tokens + out_tokens)
        self._sync_lease(unit, ctx)
        return stats

    async def _activate(self, unit: ExecutionUnit, ctx: Any, prompt_tokens: int) -> None:
        """首次推理 = submit;恢复 = restore(可能重算或搬运)。"""
        handle = self._handles.get(unit.unit_id)
        restored = None
        if handle is None:
            handle = await self.backend.submit(unit.unit_id, ctx.context_id, prompt_tokens)
            self._handles[unit.unit_id] = handle
        else:
            self.journal.append("restore_started", unit_id=unit.unit_id)
            restored = await self.backend.restore(handle)
        self._set_context(unit, ctx, ContextState.ACTIVE, handle.resident_tokens or prompt_tokens)
        self._sync_lease(unit, ctx)
        if restored is not None:
            unit.aggregate.recompute_tokens += restored.recompute_tokens
            self.journal.append(
                "restore_completed",
                unit_id=unit.unit_id,
                payload={
                    "recompute_tokens": restored.recompute_tokens,
                    "latency_ms": restored.latency_ms,
                },
            )
            recorder = getattr(self.policy, "record_restore", None)
            if recorder is not None:
                recorder(unit.unit_id, self.clock.now())

    def _set_context(self, unit: ExecutionUnit, ctx: Any, state: Any, tokens: int | None = None) -> None:
        from phaseweaver.phases import check_context_transition

        check_context_transition(ctx.state, state)
        ctx.state = state
        if tokens is not None:
            ctx.resident_tokens = tokens
        if state is ContextState.ACTIVE:
            ctx.resident_bytes = ctx.resident_tokens * self.backend.caps().bytes_per_token
        elif state is ContextState.RELEASED:
            ctx.resident_bytes = 0
        elif state is ContextState.DROPPED:
            ctx.resident_bytes = 0
        self.journal.append(
            "context_state_changed",
            unit_id=unit.unit_id,
            payload={
                "context_id": ctx.context_id,
                "state": state.value,
                "resident_bytes": ctx.resident_bytes,
                "resident_tokens": ctx.resident_tokens,
            },
        )

    def _new_context(self, unit: ExecutionUnit) -> Any:
        from phaseweaver.types import InferenceContext

        ctx = InferenceContext(context_id=f"ctx-{unit.unit_id}")
        unit.context = ctx
        return ctx

    # --- 内部:工具 -------------------------------------------------------
    async def _call_tool(
        self, unit: ExecutionUnit, spec: ToolSpec, args: dict[str, Any]
    ) -> ToolResult:
        from phaseweaver.tools.base import ToolStatus

        ctx = unit.context
        expected = spec.expected_duration_s or self.config.default_expected_tool_s
        if ctx is not None and ctx.state is ContextState.ACTIVE:
            disposition = self.policy.disposition_for(unit.unit_id, self._snapshot(), expected)
            await self._park(unit, ctx, disposition)
        if unit.phase is Phase.REASONING:
            self.registry.transition(unit.unit_id, Phase.ACTING)
        tool_unit = self.registry.create_tool_call(unit.unit_id, spec.name)
        self.registry.transition(tool_unit.unit_id, Phase.ACTING)
        try:
            result = await self.supervisor.run(tool_unit.unit_id, spec, args)
        except asyncio.CancelledError:
            if not tool_unit.phase.is_terminal:
                self.registry.transition(tool_unit.unit_id, Phase.CANCELLED)
            raise
        if not tool_unit.phase.is_terminal:
            self.registry.transition(tool_unit.unit_id, Phase.COMPLETED)
        if unit.phase is Phase.ACTING:
            if result.status in (ToolStatus.OK, ToolStatus.ERROR):
                self.registry.transition(unit.unit_id, Phase.REASONING)
            else:  # timeout / killed / cancelled:任务失败,不伪装成功
                self.registry.transition(unit.unit_id, Phase.FAILED)
        return result

    # --- 内部:处置 -------------------------------------------------------
    async def _park(self, unit: ExecutionUnit, ctx: Any, disposition: Disposition) -> None:
        from phaseweaver.types import LeaseKind

        handle = self._handles.get(unit.unit_id)
        if handle is None:
            return
        self.journal.append(
            "park_started", unit_id=unit.unit_id, payload={"disposition": disposition.value}
        )
        result = await self.backend.park(handle, disposition)
        self._set_context(unit, ctx, self._state_for(result.applied))
        if result.applied is Disposition.RETAIN:
            pass  # I2:RETAINED 必须继续持有 KV 租约
        else:
            lease = self._leases.pop(unit.unit_id, None)
            if lease is not None:
                self.broker.release(lease.lease_id, reason="park")
        if result.applied is Disposition.OFFLOAD and result.moved_bytes > 0:
            self.broker.acquire(
                unit.unit_id,
                ctx.context_id,
                LeaseKind.OFFLOAD_BYTES,
                result.moved_bytes,
                unit.priority,
            )
        self.journal.append(
            "park_completed",
            unit_id=unit.unit_id,
            payload={
                "applied": result.applied.value,
                "degraded_from": result.degraded_from.value if result.degraded_from else None,
                "moved_bytes": result.moved_bytes,
                "latency_ms": result.latency_ms,
            },
        )
        recorder = getattr(self.policy, "record_park", None)
        if recorder is not None:
            recorder(unit.unit_id, self.clock.now())
        self._detect_thrash(unit)

    def _detect_thrash(self, unit: ExecutionUnit) -> None:
        """同一单元在窗口内反复 park -> 记 thrash 事件,scheduler 会强制提档。"""
        window_s = float(getattr(getattr(self.policy, "config", None), "thrash_window_s", 60.0))
        now = self.clock.now()
        history = [ts for ts in self._park_times.get(unit.unit_id, []) if now - ts <= window_s]
        history.append(now)
        self._park_times[unit.unit_id] = history
        if len(history) >= 2:
            state = self.policy._state(unit.unit_id) if hasattr(self.policy, "_state") else None
            if state is not None:
                state.park_count_window = len(history)
            self.journal.append(
                "thrash",
                unit_id=unit.unit_id,
                payload={"count": len(history), "window_s": window_s},
            )

    @staticmethod
    def _state_for(disposition: Disposition) -> ContextState:
        return {
            Disposition.RETAIN: ContextState.RETAINED,
            Disposition.DROP: ContextState.DROPPED,
            Disposition.OFFLOAD: ContextState.OFFLOADED,
            Disposition.RELEASE: ContextState.RELEASED,
        }[disposition]

    def _sync_lease(self, unit: ExecutionUnit, ctx: Any) -> None:
        from phaseweaver.types import LeaseKind

        want = ctx.resident_bytes
        current = self._leases.get(unit.unit_id)
        if current is not None and current.amount == want and current.active:
            return
        if current is not None:
            self.broker.release(current.lease_id, reason="resize")
        lease = self.broker.acquire(
            unit.unit_id, ctx.context_id, LeaseKind.KV_BYTES, want, unit.priority
        )
        if lease is None:
            raise AdmissionRejected(f"kv budget exhausted for {unit.unit_id}")
        self._leases[unit.unit_id] = lease

    async def _release_unit(self, unit: ExecutionUnit) -> None:
        lease = self._leases.pop(unit.unit_id, None)
        if lease is not None:
            self.broker.release(lease.lease_id, reason="unit_end")
        if unit.context is not None and unit.context.state is not ContextState.RELEASED:
            self._set_context(unit, unit.context, ContextState.RELEASED)

    async def _cancel(self, unit: ExecutionUnit, reason: str) -> None:
        if not unit.phase.is_terminal:
            self.registry.transition(unit.unit_id, Phase.CANCELLED)
        await self._release_unit(unit)

    async def _fork(self, parent: ExecutionUnit, label: str, fn: AgentFn) -> TaskOutcome:
        from phaseweaver.types import ForkMode

        child = self.registry.fork(parent.unit_id, label, mode=ForkMode.SHARE)
        started = self.clock.now()
        self.registry.transition(child.unit_id, Phase.QUEUED)
        await fn(AgentHandle(self, child))
        if not child.phase.is_terminal:
            self.registry.transition(child.unit_id, Phase.COMPLETED)
        self.registry.join(child.unit_id)
        return TaskOutcome(
            unit_id=child.unit_id,
            phase=child.phase.value,
            latency_ms=(self.clock.now() - started) * 1000,
        )
```

> `MockBackend(journal=None)` 在本 Task 里只用于构造;`Runtime` 不依赖 backend 的 journal。Task 12 会统一改为由 Runtime 注入。

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/integration/test_runtime_flow.py -q`
Expected: `2 passed`

> 若报 `EngineOutOfMemory`,说明 `_sync_lease` 在 `complete()` 之后才申请租约(此时引擎已分配),检查容量是否足够;测试用的是 4 MiB 容量与 1000 B/token,足够。

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat(runtime): agent handle wiring phase, lease, disposition and tools"
```

---

## Task 11: 全局 SLO 调度器(防抖 + DRR + I8)

**Files:**
- Create: `src/phaseweaver/scheduler.py`
- Modify: `src/phaseweaver/invariant.py`(追加 I8)
- Test: `tests/unit/test_scheduler.py`, `tests/property/test_no_starvation.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_scheduler.py`**

```python
from phaseweaver.policy import ClusterSnapshot
from phaseweaver.scheduler import SchedulerConfig, SystemScheduler
from phaseweaver.types import Disposition


def snap(*, resident: int, capacity: int = 1000, slack: float = 60.0, thrash: int = 0) -> ClusterSnapshot:
    return ClusterSnapshot(
        now=100.0,
        capacity_bytes=capacity,
        resident_bytes=resident,
        queued=1,
        reasoning=1,
        slack_by_unit={"u1": slack},
        recent_thrash={"u1": thrash},
    )


def test_urgent_unit_is_retained() -> None:
    sched = SystemScheduler(SchedulerConfig())
    # slack 很小、工具很短 -> 保住
    assert sched.disposition_for("u1", snap(resident=100, slack=0.5), 1.0) is Disposition.RETAIN


def test_relaxed_unit_yields_even_for_short_tool() -> None:
    sched = SystemScheduler(SchedulerConfig())
    assert sched.disposition_for("u1", snap(resident=100, slack=600.0), 1.0) is Disposition.OFFLOAD


def test_hopeless_unit_is_not_prioritized() -> None:
    sched = SystemScheduler(SchedulerConfig())
    assert sched.disposition_for("u1", snap(resident=100, slack=-30.0), 1.0) is Disposition.DROP


def test_high_pressure_degrades_one_level() -> None:
    sched = SystemScheduler(SchedulerConfig())
    # 压力 0.95 -> 下移一档:OFFLOAD 变 DROP
    assert sched.disposition_for("u1", snap(resident=950, slack=600.0), 1.0) is Disposition.DROP


def test_thrashing_context_is_promoted() -> None:
    sched = SystemScheduler(SchedulerConfig())
    assert sched.disposition_for("u1", snap(resident=100, slack=600.0, thrash=3), 30.0) is Disposition.RETAIN


def test_cooldown_suppresses_repeated_park() -> None:
    sched = SystemScheduler(SchedulerConfig(park_cooldown_s=10.0))
    sched.record_park("u1", now=100.0)
    assert sched.disposition_for("u1", snap(resident=100, slack=600.0), 30.0) is Disposition.RETAIN
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/unit/test_scheduler.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.scheduler`

- [ ] **Step 3: 写 `src/phaseweaver/scheduler.py`**

```python
"""B2:全局 SLO 调度器(ADR-0005)。

决策只看只读快照 + 自身的历史记录(park 冷却、抖动计数、DRR 配额)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from phaseweaver.policy import ClusterSnapshot
from phaseweaver.types import Disposition

_LADDER = (Disposition.RETAIN, Disposition.OFFLOAD, Disposition.DROP)


@dataclass
class SchedulerConfig:
    retain_max_s: float = 2.0
    offload_max_s: float = 20.0
    pressure_threshold: float = 0.8
    min_residency_s: float = 5.0
    park_cooldown_s: float = 10.0
    hysteresis_band: float = 0.15
    thrash_window_s: float = 60.0
    thrash_threshold: int = 2
    quantum_slots: int = 1
    allow_offload: bool = True


@dataclass
class _UnitState:
    last_park_ts: float = -1e9
    last_restore_ts: float = -1e9
    park_count_window: int = 0
    deficit: int = 0
    resident_since: float = 0.0


@dataclass
class SystemScheduler:
    config: SchedulerConfig = field(default_factory=SchedulerConfig)
    _units: dict[str, _UnitState] = field(default_factory=dict)

    def _state(self, unit_id: str) -> _UnitState:
        return self._units.setdefault(unit_id, _UnitState())

    # --- 协议 -------------------------------------------------------------
    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return snap.pressure < 0.98

    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        st = self._state(unit_id)
        if snap.pressure >= 0.98:
            return False
        if snap.now - st.last_park_ts < self.config.park_cooldown_s:
            return False
        return True

    def disposition_for(
        self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float
    ) -> Disposition:
        cfg = self.config
        st = self._state(unit_id)

        # 1) 抖动保护:近期反复 park/restore -> 强制保留
        if st.park_count_window >= cfg.thrash_threshold:
            return Disposition.RETAIN
        # 2) 冷却:刚 park 过,不再 park
        if snap.now - st.last_park_ts < cfg.park_cooldown_s:
            return Disposition.RETAIN
        # 3) 最小驻留
        if snap.now - st.resident_since < cfg.min_residency_s and snap.pressure < cfg.pressure_threshold:
            return Disposition.RETAIN

        slack = snap.slack_by_unit.get(unit_id, 1e9)

        # 4) SLO 已不可满足:放弃赌注,不为它抢占
        if slack < 0:
            return Disposition.DROP
        # 5) 极其紧急:保住
        if slack < expected_tool_s:
            return Disposition.RETAIN
        # 6) 不着急:按基础分档,并因"富余"整体下移一档
        base = self._base(expected_tool_s)
        if slack > 2.0 * max(expected_tool_s, 1e-6):
            base = self._shift(base, +1)
        # 7) 容量压力:整体下移一档
        if snap.pressure > cfg.pressure_threshold:
            base = self._shift(base, +1)
        return self._degrade_if_needed(base)

    # --- 记账 -------------------------------------------------------------
    def record_park(self, unit_id: str, now: float) -> None:
        st = self._state(unit_id)
        st.last_park_ts = now
        st.park_count_window += 1

    def record_restore(self, unit_id: str, now: float) -> None:
        st = self._state(unit_id)
        st.last_restore_ts = now

    def tick(self, now: float) -> None:
        """推进一个时间片:抖动窗口滑动 + DRR 配额累加。"""
        for st in self._units.values():
            if now - st.last_restore_ts > self.config.thrash_window_s:
                st.park_count_window = 0
            st.deficit += self.config.quantum_slots

    # --- 内部 -------------------------------------------------------------
    def _base(self, expected_tool_s: float) -> Disposition:
        if expected_tool_s < self.config.retain_max_s:
            return Disposition.RETAIN
        if self.config.allow_offload and expected_tool_s < self.config.offload_max_s:
            return Disposition.OFFLOAD
        return Disposition.DROP

    @staticmethod
    def _shift(d: Disposition, steps: int) -> Disposition:
        idx = min(max(_LADDER.index(d) + steps, 0), len(_LADDER) - 1)
        return _LADDER[idx]

    def _degrade_if_needed(self, d: Disposition) -> Disposition:
        if d is Disposition.OFFLOAD and not self.config.allow_offload:
            return Disposition.DROP
        return d
```

- [ ] **Step 4: 写 `tests/property/test_no_starvation.py`**

```python
"""I8:DRR 配额累加保证已准入单元不会无限期饥饿。"""

from hypothesis import given, settings, strategies as st

from phaseweaver.policy import ClusterSnapshot
from phaseweaver.scheduler import SchedulerConfig, SystemScheduler


@settings(max_examples=30, deadline=None)
@given(units=st.lists(st.text(alphabet="abcdefgh", min_size=1, max_size=3), min_size=1, max_size=8, unique=True))
def test_every_admitted_unit_gets_admission_eventually(units: list[str]) -> None:
    sched = SystemScheduler(SchedulerConfig())
    snap = ClusterSnapshot(
        now=0.0,
        capacity_bytes=1000,
        resident_bytes=100,
        queued=len(units),
        reasoning=1,
        slack_by_unit={u: 100.0 for u in units},
        recent_thrash={},
    )
    for u in units:
        assert sched.admit(u, snap)
    for _ in range(10):
        sched.tick(1.0)
    assert all(sched._state(u).deficit >= 10 for u in units)
```

- [ ] **Step 5: 在 `invariant.py` 追加 I8**

```python
    def _check_i8(self) -> list[Violation]:
        out: list[Violation] = []
        for unit in self._registry.all_units():
            waited = unit.arrival_ts
            del waited  # 等待时长由 Reducer 从日志计算,此处只校验显式超限
            if unit.phase is Phase.QUEUED and unit.deadline is not None:
                if self._clock.now() > unit.deadline + 60.0:
                    out.append(Violation("I8", unit.unit_id, "queued beyond deadline + 60s"))
        return out
```

并在 `check()` 中加入 `violations.extend(self._check_i8())`,构造器增加 `clock` 参数:

```python
    def __init__(self, registry, broker, journal, *, strict=True, clock=None) -> None:
        self._clock = clock
```

无 clock 时 `_check_i8` 直接返回空列表:`if self._clock is None: return []`。

- [ ] **Step 6: 运行测试**

Run: `uv run pytest tests/unit/test_scheduler.py tests/property/test_no_starvation.py -q`
Expected: 全绿

- [ ] **Step 7: 提交**

```bash
git add -A && git commit -m "feat(scheduler): global SLO scheduler with anti-thrash knobs and DRR"
```

---

## Task 12: 指标 Reducer 与 RunReport

**Files:**
- Create: `src/phaseweaver/metrics.py`
- Test: `tests/unit/test_metrics.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_metrics.py`**

```python
from phaseweaver.clock import VirtualClock
from phaseweaver.journal import EventJournal
from phaseweaver.metrics import MetricsReducer, percentile


def build_journal() -> EventJournal:
    clock = VirtualClock()
    j = EventJournal(run_id="r1", clock=clock)
    j.append("run_started")
    j.append(
        "unit_created",
        unit_id="r1/root",
        payload={"kind": "agent", "arrival_ts": 0.0, "deadline": 100.0},
    )
    j.append("phase_transitioned", unit_id="r1/root", payload={"from": "PENDING", "to": "QUEUED"})
    j.append("phase_transitioned", unit_id="r1/root", payload={"from": "QUEUED", "to": "REASONING"})
    j.append("phase_transitioned", unit_id="r1/root", payload={"from": "REASONING", "to": "COMPLETED"})
    return j


def test_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([], 95) == 0.0


def test_report_counts_completed_task() -> None:
    report = MetricsReducer.replay(build_journal())
    assert report.completed == 1
    assert report.slo_attainment_rate == 1.0
    assert len(report.latencies_ms) == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/unit/test_metrics.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.metrics`

- [ ] **Step 3: 写 `src/phaseweaver/metrics.py`**

```python
"""从事件日志重放指标(ADR-0003:不维护独立计数器)。"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

from phaseweaver.clock import Clock
from phaseweaver.journal import EventJournal


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


@dataclass
class RunReport:
    run_id: str
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    slo_hits: int = 0
    wall_clock_s: float = 0.0
    park_count: int = 0
    thrash_count: int = 0
    recompute_tokens: int = 0
    peak_resident_bytes: int = 0
    gpu_idle_wasted_bytes_s: float = 0.0
    admission_rejects: int = 0
    restore_latencies_ms: list[float] = field(default_factory=list)
    queue_waits_ms: list[float] = field(default_factory=list)

    @property
    def slo_attainment_rate(self) -> float:
        total = self.completed + self.failed + self.cancelled
        return (self.slo_hits / total) if total else 0.0

    @property
    def goodput(self) -> float:
        return (self.slo_hits / self.wall_clock_s) if self.wall_clock_s > 0 else 0.0

    @property
    def latency_p50_ms(self) -> float:
        return percentile(self.latencies_ms, 50)

    @property
    def latency_p95_ms(self) -> float:
        return percentile(self.latencies_ms, 95)

    @property
    def latency_p99_ms(self) -> float:
        return percentile(self.latencies_ms, 99)

    @property
    def latency_cv(self) -> float:
        if len(self.latencies_ms) < 2:
            return 0.0
        m = mean(self.latencies_ms)
        return (pstdev(self.latencies_ms) / m) if m else 0.0

    @property
    def thrash_rate(self) -> float:
        minutes = self.wall_clock_s / 60.0
        return (self.thrash_count / minutes) if minutes > 0 else 0.0

    def summary_row(self) -> dict[str, float]:
        return {
            "completed": float(self.completed),
            "slo_rate": round(self.slo_attainment_rate, 4),
            "goodput": round(self.goodput, 4),
            "p50_ms": round(self.latency_p50_ms, 1),
            "p95_ms": round(self.latency_p95_ms, 1),
            "p99_ms": round(self.latency_p99_ms, 1),
            "lat_cv": round(self.latency_cv, 3),
            "thrash": round(self.thrash_rate, 2),
            "recompute_tok": float(self.recompute_tokens),
            "peak_kv_mib": round(self.peak_resident_bytes / 1024 / 1024, 1),
            "idle_waste_mib_s": round(self.gpu_idle_wasted_bytes_s / 1024 / 1024, 1),
        }


class MetricsReducer:
    @staticmethod
    def replay(journal: EventJournal, *, clock: Clock | None = None) -> RunReport:
        report = RunReport(run_id=journal.run_id)
        arrivals: dict[str, tuple[float, float]] = {}
        resident: dict[str, int] = {}
        reasoning_units: set[str] = set()
        last_ts = 0.0
        idle_accum = 0.0

        def resident_total() -> int:
            return sum(resident.values())

        for ev in journal.events():
            dt = max(0.0, ev.ts - last_ts)
            if not reasoning_units:
                idle_accum += dt * resident_total()
            last_ts = ev.ts

            if ev.event == "unit_created":
                payload = ev.payload
                is_task_unit = payload.get("kind") in ("agent", "subagent")
                if ev.unit_id and is_task_unit and "arrival_ts" in payload:
                    arrivals[ev.unit_id] = (
                        float(payload["arrival_ts"]),
                        float(payload.get("deadline") or 0.0),
                    )
            elif ev.event == "phase_transitioned" and ev.unit_id:
                to = str(ev.payload.get("to", ""))
                if to == "REASONING":
                    reasoning_units.add(ev.unit_id)
                elif to in ("ACTING", "WAITING", "COMPLETED", "FAILED", "CANCELLED"):
                    reasoning_units.discard(ev.unit_id)
                if to in ("COMPLETED", "FAILED", "CANCELLED") and ev.unit_id in arrivals:
                    arrived, deadline = arrivals[ev.unit_id]
                    latency_ms = (ev.ts - arrived) * 1000
                    report.latencies_ms.append(latency_ms)
                    hit = latency_ms <= deadline * 1000 if deadline > 0 else True
                    if to == "COMPLETED":
                        report.completed += 1
                        if hit:
                            report.slo_hits += 1
                    elif to == "FAILED":
                        report.failed += 1
                    else:
                        report.cancelled += 1
            elif ev.event == "context_state_changed" and ev.unit_id:
                resident[ev.unit_id] = int(ev.payload.get("resident_bytes", 0))
                report.peak_resident_bytes = max(report.peak_resident_bytes, resident_total())
            elif ev.event == "park_completed":
                report.park_count += 1
            elif ev.event == "thrash":
                report.thrash_count += 1
            elif ev.event == "admission_rejected":
                report.admission_rejects += 1
            elif ev.event == "restore_completed":
                report.restore_latencies_ms.append(float(ev.payload.get("latency_ms", 0.0)))
                report.recompute_tokens += int(ev.payload.get("recompute_tokens", 0))
            elif ev.event == "run_finished":
                pass

        report.wall_clock_s = last_ts
        report.gpu_idle_wasted_bytes_s = idle_accum
        return report
```

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/unit/test_metrics.py -q`
Expected: `2 passed`

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat(metrics): journal replay reducer with tail-latency and thrash metrics"
```

---

## Task 13: 工作负载、实验 CLI 与 E1

**Files:**
- Create: `examples/workloads/w1.toml`, `examples/workloads/w2.toml`, `examples/workloads/w4.toml`,
  `src/phaseweaver/demo.py`
- Test: `tests/integration/test_demo_compare.py`

- [ ] **Step 1: 写 `examples/workloads/w1.toml`**

```toml
[workload]
name = "w1"
agents = 12
rounds = 4
context_tokens = 2048
out_tokens = 128
slo_s = 240.0
seed = 7

[tool]
kind = "python"
p50_s = 2.0
sigma = 0.6

[arrival]
kind = "poisson"
rate_per_s = 0.25
```

- [ ] **Step 2: 写 `src/phaseweaver/demo.py`(只实现 `compare`,E1)**

```python
"""实验 CLI。所有表格必须带 [L1]/[L2] 标头(规格 §12.5)。"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from phaseweaver.metrics import MetricsReducer
from phaseweaver.policy import PinAlwaysPolicy, Scheduler, TtlEvictPolicy, YieldOnToolPolicy
from phaseweaver.runtime import Runtime, RuntimeConfig
from phaseweaver.tools.base import ToolSpec


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

    @classmethod
    def load(cls, path: Path) -> "Workload":
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
        )


def caps(capacity_gib: float = 8.0) -> BackendCaps:
    return BackendCaps(
        capacity_bytes=int(capacity_gib * 1024**3),
        block_tokens=16,
        bytes_per_token=57344,
        prefill_tok_per_s=3200,
        decode_tok_per_s=45,
        mem_bw_bps=int(12 * 1024**3),
        supports_offload=True,
    )


def build_policy(name: str) -> Scheduler:
    if name == "B0":
        return PinAlwaysPolicy()
    if name == "B1":
        return TtlEvictPolicy(ttl_s=5.0)
    if name == "B2a":
        return YieldOnToolPolicy()
    raise ValueError(f"unknown policy {name}")


async def run_one(workload: Workload, policy_name: str, *, capacity_gib: float = 8.0) -> Any:
    clock = VirtualClock()
    clock.start_driver()
    backend = MockBackend(caps=caps(capacity_gib), clock=clock, journal=None)  # type: ignore[arg-type]
    rt = Runtime(
        backend=backend,
        clock=clock,
        budget=Budget(total_bytes=caps(capacity_gib).capacity_bytes, headroom_frac=0.0),
        policy=build_policy(policy_name),
        config=RuntimeConfig(run_id=f"{workload.name}-{policy_name}"),
    )
    await rt.start()
    rng = random.Random(workload.seed)
    tools = [
        ToolSpec(
            f"tool-{i}",
            "python",
            expected_duration_s=max(0.1, rng.lognormvariate(0.0, workload.tool_sigma) * workload.tool_p50_s),
        )
        for i in range(workload.rounds)
    ]

    async def agent(handle: Any) -> None:
        for spec in tools:
            await handle.reason(prompt_tokens=workload.context_tokens, out_tokens=workload.out_tokens)
            await handle.call_tool(spec, {"code": "pass"})

    pending: list[asyncio.Task[Any]] = []
    for i in range(workload.agents):
        if workload.rate_per_s > 0:
            await clock.wait(rng.expovariate(workload.rate_per_s))
        pending.append(asyncio.create_task(rt.run_task(f"task-{i}", agent, slo_s=workload.slo_s)))
    await asyncio.gather(*pending)
    await rt.shutdown()
    return MetricsReducer.replay(rt.journal)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phaseweaver")
    sub = parser.add_subparsers(dest="cmd", required=True)
    cmp_p = sub.add_parser("compare", help="B0/B1/B2a 对比(E1)")
    cmp_p.add_argument("--workload", required=True)
    cmp_p.add_argument("--policies", default="B0,B1,B2a")
    cmp_p.add_argument("--capacity-gib", type=float, default=8.0)
    args = parser.parse_args(argv)

    workload = Workload.load(Path(args.workload))
    print(f"[L1] {workload.name} capacity={args.capacity_gib}GiB slo={workload.slo_s}s")
    print(f"{'policy':<8}{'done':>6}{'slo':>8}{'goodput':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'thrash':>8}{'recomp':>9}")
    rows = []
    for policy in args.policies.split(","):
        report = asyncio.run(run_one(workload, policy, capacity_gib=args.capacity_gib))
        row = report.summary_row()
        rows.append({"policy": policy, **row})
        print(
            f"{policy:<8}{int(row['completed']):>6}{row['slo_rate']:>8.3f}{row['goodput']:>9.3f}"
            f"{row['p50_ms']:>9.0f}{row['p95_ms']:>9.0f}{row['p99_ms']:>9.0f}"
            f"{row['thrash']:>8.2f}{int(row['recompute_tok']):>9}"
        )
    print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: 写测试 `tests/integration/test_demo_compare.py`**

```python
from pathlib import Path

from phaseweaver.demo import Workload, run_one
import asyncio


def test_workload_loads() -> None:
    w = Workload.load(Path("examples/workloads/w1.toml"))
    assert w.agents == 12


def test_compare_runs_and_reports() -> None:
    w = Workload.load(Path("examples/workloads/w1.toml"))
    report = asyncio.run(run_one(w, "B0"))
    assert report.completed >= 1
    assert report.wall_clock_s > 0
```

- [ ] **Step 4: 运行**

Run: `uv run pytest tests/integration/test_demo_compare.py -q`
Expected: `2 passed`

> 若耗时超过 60 秒,说明虚拟时钟没有推进(所有协程都在等真实时间)。检查 `MockBackend.complete` 是否走了 `await self._clock.wait(...)`。

- [ ] **Step 5: 手工跑一次 E1**

Run: `uv run python -m phaseweaver.demo compare --workload examples/workloads/w1.toml`
Expected: 三行 B0/B1/B2a 对比表 + 一行 JSON。把输出贴进 `docs/notes/experiments.md`。

- [ ] **Step 6: 提交**

```bash
git add -A && git commit -m "feat(exp): workload loader and E1 comparison CLI"
```

---

## Task 14: 实验矩阵 E2–E4(突发、多 seed、容量扫描)

**Files:**
- Modify: `src/phaseweaver/demo.py`(新增 `bench` / `sweep` / `seeds` 子命令)
- Create: `tests/integration/test_sweep.py`

- [ ] **Step 1: 给 `demo.py` 加子命令**

```python
def cmd_sweep(args: argparse.Namespace) -> int:
    workload = Workload.load(Path(args.workload))
    print(f"[L2] sweep {workload.name} seeds={args.seeds}")
    for gib in [float(x) for x in args.capacities.split(",")]:
        for policy in args.policies.split(","):
            acc: list[float] = []
            for seed in range(args.seeds):
                workload.seed = seed
                report = asyncio.run(run_one(workload, policy, capacity_gib=gib))
                acc.append(report.goodput)
            mean = sum(acc) / len(acc)
            var = sum((x - mean) ** 2 for x in acc) / len(acc)
            print(f"C={gib:<5} {policy:<4} goodput={mean:.4f} std={var ** 0.5:.4f}")
    return 0


def cmd_burst(args: argparse.Namespace) -> int:
    """E2:把 arrival rate 在中途放大 3 倍,观察 p99 与恢复时间。"""
    workload = Workload.load(Path(args.workload))
    base_rate = workload.rate_per_s
    for policy in args.policies.split(","):
        workload.rate_per_s = base_rate * 3
        report = asyncio.run(run_one(workload, policy, capacity_gib=args.capacity_gib))
        print(f"[L1] burst {policy}: p99={report.latency_p99_ms:.0f}ms "
              f"slo={report.slo_attainment_rate:.3f} thrash={report.thrash_rate:.2f}")
    return 0
```

并在 `main()` 中注册:

```python
    sw = sub.add_parser("sweep")
    sw.add_argument("--workload", required=True)
    sw.add_argument("--capacities", default="2,4,8,16")
    sw.add_argument("--policies", default="B0,B1,B2a")
    sw.add_argument("--seeds", type=int, default=20)
    bu = sub.add_parser("burst")
    bu.add_argument("--workload", required=True)
    bu.add_argument("--policies", default="B0,B1,B2a")
    bu.add_argument("--capacity-gib", type=float, default=8.0)
```

以及分发:

```python
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "sweep":
        return cmd_sweep(args)
    if args.cmd == "burst":
        return cmd_burst(args)
    raise SystemExit(f"unknown command {args.cmd}")
```

> 把原 `main()` 里的 compare 逻辑抽成 `cmd_compare(args)` 函数,便于分发。

- [ ] **Step 2: 写测试 `tests/integration/test_sweep.py`**

```python
import asyncio

from phaseweaver.demo import Workload, run_one


def test_multi_seed_is_deterministic_for_same_seed() -> None:
    w = Workload.load(__import__("pathlib").Path("examples/workloads/w1.toml"))
    a = asyncio.run(run_one(w, "B2a"))
    b = asyncio.run(run_one(w, "B2a"))
    assert a.summary_row() == b.summary_row()
```

- [ ] **Step 3: 运行**

Run: `uv run pytest tests/integration/test_sweep.py -q`
Expected: `1 passed`(若不稳定,说明虚拟时钟或随机数源不确定,必须先修确定性再继续)

- [ ] **Step 4: 跑 E2/E3/E4 并把输出写入 `docs/notes/experiments.md`**

```bash
uv run python -m phaseweaver.demo burst --workload examples/workloads/w1.toml
uv run python -m phaseweaver.demo sweep --workload examples/workloads/w1.toml --seeds 20
```

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat(exp): burst, capacity sweep and multi-seed stability runs"
```

---

## Task 15: B4 离线最优上界与 E5

**Files:**
- Create: `src/phaseweaver/oracle.py`
- Modify: `src/phaseweaver/demo.py`(新增 `optimal` 子命令)
- Test: `tests/unit/test_oracle.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_oracle.py`**

```python
from phaseweaver.oracle import OracleBudget, brute_force_min_makespan


def test_oracle_finds_optimal_split() -> None:
    # 2 个任务,每个 1 次推理(5s) + 1 次工具(10s),容量只够 1 个上下文
    # 最优:先跑 T1 推理 -> T1 工具期跑 T2 推理 -> T2 工具 -> T1 推理 ...
    makespan = brute_force_min_makespan(
        tasks=[1, 1],
        reasoning_s=[5.0, 5.0],
        tool_s=[10.0, 10.0],
        budget=OracleBudget(max_resident=1, max_nodes=10_000),
    )
    assert 15.0 <= makespan <= 40.0


def test_oracle_unbounded_capacity_equals_serial_work() -> None:
    makespan = brute_force_min_makespan(
        tasks=[1], reasoning_s=[5.0], tool_s=[10.0], budget=OracleBudget(max_resident=2)
    )
    assert makespan == 15.0
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/unit/test_oracle.py -q`
Expected: FAIL `ModuleNotFoundError: phaseweaver.oracle`

- [ ] **Step 3: 写 `src/phaseweaver/oracle.py`**

```python
"""B4:离线最优上界。只在实验路径使用,禁止被 runtime 引用。"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class OracleBudget:
    max_resident: int
    max_nodes: int = 200_000


def brute_force_min_makespan(
    tasks: list[int],
    reasoning_s: list[float],
    tool_s: list[float],
    budget: OracleBudget,
) -> float:
    """枚举所有任务的全序,模拟"上下文数不超过 max_resident"的最优执行。

    简化模型:每个任务由 rounds 次(reasoning -> tool)组成;同任务的推理必须串行,
    工具与推理可重叠;任意时刻驻留上下文数 <= max_resident。
    返回该序下的完成时间;对所有序取最小。
    """
    n = len(tasks)
    best = float("inf")
    nodes = 0
    for order in itertools.permutations(range(n)):
        nodes += 1
        if nodes > budget.max_nodes:
            break
        resident: list[int] = []
        free_at: list[float] = [0.0] * n
        cursor = 0.0
        ok = True
        for idx in order:
            rounds = tasks[idx]
            t = cursor
            for _ in range(rounds):
                if len(resident) >= budget.max_resident and resident:
                    # 释放最早完成的那一个
                    resident.pop(0)
                resident.append(idx)
                t += reasoning_s[idx]
                t += tool_s[idx]
            free_at[idx] = t
            cursor = min(free_at[i] for i in order[: len(free_at)])
        makespan = max(free_at)
        best = min(best, makespan)
    return best
```

> 说明:这是**下界近似**(忽略工具与推理重叠的最优调度细节),报告里必须标注 `bound_kind=heuristic_lower_bound`。不得声称是 strict optimum。

- [ ] **Step 4: 加 `optimal` 子命令**

```python
def cmd_optimal(args: argparse.Namespace) -> int:
    from phaseweaver.oracle import OracleBudget, brute_force_min_makespan

    workload = Workload.load(Path(args.workload))
    report = asyncio.run(run_one(workload, args.policy, capacity_gib=args.capacity_gib))
    bound_s = brute_force_min_makespan(
        tasks=[workload.rounds] * workload.agents,
        reasoning_s=[workload.context_tokens / 3200 + workload.out_tokens / 45] * workload.agents,
        tool_s=[workload.tool_p50_s] * workload.agents,
        budget=OracleBudget(max_resident=max(1, int(args.capacity_gib * 128 / 2048))),
    )
    observed = report.wall_clock_s
    gap = (observed - bound_s) / bound_s * 100 if bound_s else 0.0
    print(f"[L2] optimality gap policy={args.policy} observed={observed:.1f}s "
          f"bound={bound_s:.1f}s gap={gap:.1f}% bound_kind=heuristic_lower_bound")
    return 0
```

- [ ] **Step 5: 运行**

Run: `uv run pytest tests/unit/test_oracle.py -q`
Expected: `2 passed`

- [ ] **Step 6: 提交**

```bash
git add -A && git commit -m "feat(oracle): offline lower bound and optimality gap report"
```

---

## Task 16: 收尾——导入图测试、README、第一份实验记录

**Files:**
- Create: `tests/unit/test_import_graph.py`, `docs/notes/experiments.md`
- Modify: `README.md`

- [ ] **Step 1: 导入图测试**

```python
import ast
from pathlib import Path

SRC = Path("src/phaseweaver")

FORBIDDEN = {
    "phases.py": {"runtime", "scheduler", "backends", "tools", "oracle"},
    "leases.py": {"runtime", "scheduler", "backends", "tools", "oracle"},
    "metrics.py": {"runtime", "scheduler", "backends", "tools"},
    "invariant.py": {"runtime", "scheduler", "backends", "tools", "oracle"},
    "runtime.py": {"oracle"},
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[-1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[-1])
    return names


def test_module_dependency_rules() -> None:
    for filename, banned in FORBIDDEN.items():
        mods = imported_modules(SRC / filename)
        assert not (mods & banned), f"{filename} illegally imports {mods & banned}"
```

- [ ] **Step 2: 跑全量**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy src/phaseweaver`
Expected: 全绿。若 mypy 报 `runtime.py` 的 `Any`,在该文件顶部加 `# mypy: disable-error-code=...` 并**在 README 记录技术债**。

- [ ] **Step 3: 写 `docs/notes/experiments.md`(贴真实跑出的三张表,带 L1/L2 标头)**

必须包含:环境(CPU/无 GPU)、seed 数、容量、负载参数、三张表(E1 对比、E2 突发、E3 容量扫描)、一段"结论与未成立项"。

- [ ] **Step 4: 更新 README 的"当前状态"**

```markdown
## 当前状态

- M0–M3 完成:执行单元树、双状态机、租约/处置、全局 SLO 调度、mock 仿真、E1–E5 实验。
- 下一步:M5 真实引擎适配(需要 NVIDIA GPU)。
- 已知技术债:(逐条列出本计划的偏差与 mypy 例外)
```

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "docs: experiment record, import-graph guard and status update"
```

---

## 自检清单(实现者在 M3 结束时逐条确认)

- [ ] 规格 §5.2 转移表全部有测试覆盖,且非法转移必被拒
- [ ] I1–I4、I6、I8 均有 property/unit 测试
- [ ] `uv run python -m phaseweaver.demo compare --workload examples/workloads/w1.toml` 能在本机 3 分钟内出表
- [ ] 同一 seed 两次运行 `summary_row()` 完全一致(确定性)
- [ ] 每个 Task 一次提交,历史可回溯
- [ ] `docs/notes/experiments.md` 的三张表都有 `[L1]`/`[L2]` 标头与 seed/C/负载脚注
- [ ] 没有任何文档把 L1/L2 称为真实 GPU 测量
