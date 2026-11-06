# Contributing

## 开发环境

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src/phaseweaver
```

Windows 本机有 5 条 sandbox 约束(管道/`taskkill`/`mkdtemp` 等),先读 [README](README.md#本机环境注意事项sandbox),
否则会重复踩坑。一键检查脚本:`pwsh scripts/check.ps1`。

## 工作流:规格 → 计划 → 实现

1. **改行为之前先改文档**。设计取舍记录在 `docs/adr/`,系统契约记录在 `docs/specs/`。
2. 一份规格只覆盖一个可独立交付的子系统;跨子系统请拆成多份。
3. 实现计划放 `docs/plans/`,任务粒度做到「一次提交能完成、能用一条命令验证」。
4. 每个 Task 一次提交,提交信息用 `type(scope): summary`,`type ∈ {feat,fix,test,docs,chore,refactor,perf}`。

## 不可协商的工程约束

- **不变量必须可断言**:I1–I8 在 `src/phaseweaver/invariant.py` 中定义,任何状态变更都要保持。
  改状态机前先读 [ADR-0001](docs/adr/0001-phase-context-orthogonality.md)。
- **事件日志是唯一事实来源**:禁止在运行时维护与日志并行的可变计数器([ADR-0003](docs/adr/0003-event-journal-first.md))。
  新增事件类型必须先在 `docs/specs/...§10` 的事件表登记。
- **指标只从日志重放**:`MetricsReducer.replay()` 是唯一入口。
- **数字分级**:[L1] mock 机制证明、[L2] 参数敏感性、[L3] 真实 GPU。禁止把 L1/L2 写成 L3,
  禁止只报均值不报 p99 与重算成本。

## 测试约定

- `tests/unit/`:纯函数与状态机,无 I/O。
- `tests/property/`:hypothesis,不变量与边界。
- `tests/integration/`:真实子进程、Runtime 端到端。
- 不依赖 pytest 的 `tmp_path`(本机 sandbox 下不可用),临时文件放 `runs/`。

## 评审清单

- [ ] 是否引入新的不变量违反路径?是否补了 property test?
- [ ] 事件类型是否登记?payload 是否含单位(字节/毫秒)?
- [ ] 指标口径是否变化?若变化,`docs/notes/experiments.md` 是否同步?
- [ ] 是否把 mock 数字写成了性能结论?
- [ ] 单文件是否超过 400 行?超了是否拆分?

## 提交前

```bash
pwsh scripts/check.ps1
```
