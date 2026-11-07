# 运行时完善与优化路线图

- 建立:2026-11-07
- 范围:Phase 1 的运行时内核与工程能力;不含第二阶段(durable)与真实 GPU 验证(单列)
- 原则:**每个 R 项必须有可执行的完成判据**,没有判据的条目不开工

## 现状快照

| 维度 | 状态 |
|---|---|
| 执行模型 | 单元树 + 双状态机 + I1–I8 不变量,71→87 测试 |
| 资源 | 处置权租约、原子 handoff、ISOLATE 配额、上下文压缩 |
| 调度 | B0/B1/B2a/B2/autopilot,可证下界 gap 3.7%–8.4% |
| 可观测 | 事件日志 + 重放指标(延迟/吞吐/KV 浪费/并发/公平) |
| 扩展 | `InferenceBackend` / `Scheduler` / `ToolExecutor` / `Profile` 四个协议 |
| 工程 | CI 四格、mypy 双平台、ruff、生命周期契约、后端超时、故障注入 |

## R1 状态与资源所有权(本轮完成)

- [x] `UnitLedger`:句柄/租约/park 历史收敛到一个对象(消灭三份平行字典)
- [x] `run_many` 支持**分批到达**(Poisson)与 **max_inflight 背压**
- 判据:测试覆盖到达时间单调递增、串行 vs 并行的墙钟比 > 1.8

## R2 故障注入(本轮完成)

- [x] 工具失败不致命(错误回传给 Agent 继续推理)
- [x] 任务取消后零租约泄漏、上下文全部 RELEASED
- [x] 3 个 seed 的严格不变量回归(无泄漏)
- 判据:全部为断言"清理属性",不是断言"没崩"

## R3 拆分 `runtime.py`(下一项)

现状 644 行,超过自定 400 行纪律。**必须先 R1 把状态收敛**,否则拆分只会把 K15 的
"分散状态"从文件内搬到文件间。

- `handoff.py`:`handoff_to_child` / `handoff_back` 两个自由函数,入参是 `(ledger, broker, backend, …)`
- `parking.py`:`park / restore / sync_lease`(依赖 ledger + broker)
- `runtime.py` 只留:生命周期、run_task、reason、call_tool、fork 编排
- 判据:`runtime.py` ≤ 350 行;`handoff.py` 有独立单测且能脱离 Runtime 构造;
  全部 87+ 测试与场景矩阵数字**逐位不变**

## R4 可观测性(面向排障)

- 导出 per-unit 相位时间线(Gantt JSON):`unit_id, phase, start, end`
- 增加 `trace_summary()`:每次 park/restore/handoff 的因果链(基于 `cause_seq`)
- 判据:给定一次运行,能在不读代码的情况下回答"这个任务为什么慢了"

## R5 扩展生态

- `[project.entry-points."phaseweaver.backends"]` 等入口点,第三方包可安装
- 提供一个"第三方风格"的示例包(`examples/plugin_backend/`),只依赖公开协议
- 判据:`pip install -e examples/plugin_backend` 后 `PhaseWeaver(backend=...)` 可用,
  且不 import 任何内部模块

## R6 调度与策略(研究性,有下界兜底)

- 相位对齐(K13 重开):当前占空比上限是结构性的;若任务异构(长短工具混合),
  错峰可能抬高平均并发 → 用下界衡量收益
- 判据:在混合工作负载上 goodput 提升 ≥ 5% 且 p99 不劣化,否则记录负结果并停止

## R7 真实引擎验证(依赖外部条件)

- issue #3:真实 vLLM 上跑 L3;适配器已就绪,运行时不需要改动
- 判据:一张 `[L3]` 表 + 一次能力降级记录

## 不做的

- 不做多机/分布式调度(超出单机范围)
- 不做 Agent 编排 DSL(与"运行时"定位冲突)
- 不为了让数字好看去调参;负结果照写
