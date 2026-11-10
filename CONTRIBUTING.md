# Contributing

## 开发环境

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src/phaseweaver
```

Windows 本机有 5 条 sandbox 约束(管道/`taskkill`/`mkdtemp` 等),先读 [README_cn](README_cn.md#sandbox) / [README](README.md#sandbox),
否则会重复踩坑。一键检查脚本:`pwsh scripts/check.ps1`。

## 工作流:规格 → 计划 → 实现

1. **改行为之前先改文档**。系统契约记录在 `docs/specs/`;实验、开发与取舍记录不随仓库提交(见 [AGENTS.md](AGENTS.md))。
2. 一份规格只覆盖一个可独立交付的子系统;跨子系统请拆成多份。
3. 实现计划放 `docs/plans/`,任务粒度做到「一次提交能完成、能用一条命令验证」。
4. 每个 Task 一次提交,提交信息用 `type(scope): summary`,`type ∈ {feat,fix,test,docs,chore,refactor,perf}`。

## 不可协商的工程约束

- **不变量必须可断言**:I1–I8 在 `src/phaseweaver/invariant.py` 中定义,任何状态变更都要保持。
  改状态机前先读设计规格中「阶段与上下文正交」一节。
- **事件日志是唯一事实来源**:禁止在运行时维护与日志并行的可变计数器。
  新增事件类型必须先在 `docs/specs/...§10` 的事件表登记。
- **指标只从日志重放**:`MetricsReducer.replay()` 是唯一入口。
- **数字分级**:[L1] mock 机制证明、[L2] 参数敏感性、[L3] 真实 GPU。禁止把 L1/L2 写成 L3,
  禁止只报均值不报 p99 与重算成本。

## 扩展点

四个协议构成对外扩展面,实现它们不需要改运行时内核:

| 想做什么 | 实现 | 参考 |
|---|---|---|
| 接一个新推理引擎 | `InferenceBackend` | `backends/mock.py`、`backends/openai_compat.py` |
| 换一种资源策略 | `Scheduler` | `policy.py`、`scheduler.py`、`autopilot.py` |
| 加一种工具 | `ToolExecutor` | `tools/proc.py` |
| 定义一个新场景 | `Profile` | `profile.py` |

完整步骤与约束见 [扩展指南](docs/guide/extending.md)。新增场景必须附一个可复现工作负载与
一组实验数字,否则视为未完成。

## API 稳定性

- 公开面是 `phaseweaver/__init__.py` 的 `__all__`;`tests/unit/test_app.py` 会守住它。
- 删除或重命名 `__all__` 中的符号属于 **breaking change**,需要 major 版本 + CHANGELOG 记录。
- `phaseweaver.backends.*`、`phaseweaver.tools.*` 视为半公开:可以扩展,但不保证跨 minor 版本
  的构造函数兼容。
- 内部模块(`runtime` 之外的私有方法、`_` 前缀)随时可变。

## 发布流程

1. 更新 `CHANGELOG.md` 的 `[Unreleased]` 段。
2. 语义化版本:新增能力 → minor;修复 → patch;破坏 API/语义 → major。
3. 打 tag 前必须:`pytest` / `ruff` / `mypy`(--platform linux 与 win32)全绿,CI 四格通过。

## 测试约定

- `tests/unit/`:纯函数与状态机,无 I/O。
- `tests/property/`:hypothesis,不变量与边界。
- `tests/integration/`:真实子进程、Runtime 端到端。
- 不依赖 pytest 的 `tmp_path`(本机 sandbox 下不可用),临时文件放 `runs/`。

### 跨平台类型检查

CI 会在 ubuntu 与 windows 上分别跑 mypy,因此平台专属符号(`ctypes.WinDLL`、
`subprocess.CREATE_NEW_PROCESS_GROUP`、`os.killpg` 等)必须用 `getattr` 取。
提交前本地预演两个平台:

```bash
uv run mypy --platform linux src/phaseweaver
uv run mypy --platform win32 src/phaseweaver
```

## 评审清单

- [ ] 是否引入新的不变量违反路径?是否补了 property test?
- [ ] 事件类型是否登记?payload 是否含单位(字节/毫秒)?
- [ ] 指标口径是否变化?若变化,是否已在 PR 描述中给出复现结果(实验记录不入库)?
- [ ] 是否把 mock 数字写成了性能结论?
- [ ] 单文件是否超过 400 行?超了是否拆分?

## 提交前

```bash
pwsh scripts/check.ps1
```
