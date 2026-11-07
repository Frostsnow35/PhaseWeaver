# 扩展指南

PhaseWeaver 的扩展点是**协议**,不是基类。实现协议即可接入,不需要改运行时。

## 1. 接一个新推理引擎(`InferenceBackend`)

```python
from phaseweaver.backends.base import BackendCaps, GenStats, Handle, ParkResult, RestoreResult
from phaseweaver.types import Disposition

class MyEngine:
    def caps(self) -> BackendCaps: ...          # 容量、块大小、每 token 字节、能力位
    def available_bytes(self) -> int: ...       # 引擎真实可用量(比字节预算更紧时以它为准)
    async def submit(self, unit_id, context_id, tokens) -> Handle: ...
    async def complete(self, handle, out_tokens) -> GenStats: ...
    async def abort(self, handle, reason) -> None: ...
    async def park(self, handle, disposition) -> ParkResult: ...
    async def restore(self, handle) -> RestoreResult: ...
    async def handoff(self, handle, new_unit_id, new_context_id) -> Handle: ...
    async def reserve(self, handle, tokens) -> None: ...
    async def truncate(self, handle, tokens) -> int: ...
```

**必须诚实声明能力**:`supports_offload=False` 时,运行时会把 `OFFLOAD` 降级为 `DROP` 并记录
`degraded_from`。假装支持会让调度器做出无法执行的决策。

**参考实现**:`backends/mock.py`(完整语义)、`backends/openai_compat.py`(HTTP,只会 `DROP`)、
`backends/vllm.py`(可选引擎级 sleep/wake)。

## 2. 写一个新策略(`Scheduler`)

```python
from phaseweaver.policy import ClusterSnapshot
from phaseweaver.types import Disposition

class MyPolicy:
    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool: ...
    def disposition_for(self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float) -> Disposition: ...
    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool: ...
```

约束:

- 策略**只能读快照**,不能改运行时状态;所有变更由 `Runtime` 执行。
- 必须尊重"SLO 已不可达时不为它抢占他人"这类安全规则(参考 `scheduler.py` 的实现顺序)。
- 若实现了 `record_park` / `record_restore` / `note_activated`,运行时会调用,用于学习/统计。

**参考实现**:`policy.py`(B0/B1/B2a)、`scheduler.py`(B2)、`autopilot.py`(压力自适应)。

## 3. 加一种工具(`ToolExecutor`)

```python
class MyTool:
    name = "mytool"
    async def run(self, args: dict, spec: ToolSpec) -> ToolResult: ...
```

要点:

- **必须可取消**:收到 `CancelledError` 时要终止真实进程,不能留下孤儿(参考 `tools/proc.py`
  的 Job Object / 进程组实现)。
- 必须返回统一信封 `ToolResult(status=ok|error|timeout|cancelled|killed, ...)`,
  运行时据此决定任务是否失败。
- 在 `tools/supervisor.py` 的注册表里登记(C 层扩展时更推荐用 entry point,见 §5)。

## 4. 定义一个新场景(`Profile`)

```python
from phaseweaver.profile import Profile

CHAT = Profile(
    name="chat",
    description="低延迟对话:短工具、SLO 很紧",
    policy="B2",
    inherit_mode="none",
    headroom_frac=0.10,
    slo_s=30.0,
)
```

Profile 只是默认值的集合,不影响运行时能力。新增场景时请同时给出一个可复现工作负载
(`examples/workloads/*.toml`),并在 `docs/notes/experiments.md` 里记录一组数字——没有数字的
场景档案等于没有验证。

## 5. 模块化扩展(入口点,已实现)

三类扩展都可以**独立发包**,不需要改运行时源码:

```toml
[project.entry-points."phaseweaver.backends"]
my-engine = "my_pkg.engine:build_backend"      # 零参工厂 -> InferenceBackend

[project.entry-points."phaseweaver.policies"]
my-policy = "my_pkg.policy:MyPolicy"           # 可调用 -> Scheduler 实例

[project.entry-points."phaseweaver.tools"]
my-tool = "my_pkg.tools:MY_EXECUTOR"           # ToolExecutor 实例
```

使用:

```python
async with PhaseWeaver(backend="my-engine", profile="research") as pw:
    await pw.run("task", agent)
```

- 入口点**懒加载**:`import phaseweaver` 不会加载任何插件;
- 插件加载失败会抛 `PluginError`,信息里包含**是哪个插件**、**哪个组**、原始异常;
- 名字冲突时**内置策略优先**(`build_policy("B2")` 永远解析到内置实现)。

完整可运行示例:[examples/plugin_backend/](../../examples/plugin_backend/)(`pip install -e .` 后
`PhaseWeaver(backend="example-device")` 即可)。

## 开发约定(摘要)

- 改状态机前先读 `docs/adr/0001-phase-context-orthogonality.md`;新增转移必须同时补测试。
- 不变量 I1–I8 见 `src/phaseweaver/invariant.py`;任何新路径都要能通过 property test。
- 事件类型是封闭集合,新增前必须登记到规格 §10。
- 提交前:`pwsh scripts/check.ps1`(或 `uv run pytest && uv run ruff check . && uv run mypy src/phaseweaver`)。
- 跨平台类型检查:`uv run mypy --platform linux src/phaseweaver` 与 `--platform win32`。
