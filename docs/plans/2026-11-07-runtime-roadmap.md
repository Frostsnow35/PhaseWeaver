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

## R3 拆分 `runtime.py`(已完成)

`runtime.py` 由 644 行降到 **320 行**,按职责拆成四个 mixin:

- [x] `parking.py`:park / restore / sync_lease / thrash 记账
- [x] `handoff.py`:交接(唯一 await + 同步转移租约与句柄)
- [x] `cleanup.py`:树形释放、取消、shield 保护
- [x] `turns.py`:reason / activate / call_tool
- 判据:`pytest` 全绿、mypy 双平台通过、同一工作负载数字不变(w1/2GiB B2 = 0.1012)

## R4 可观测性(已完成)

- [x] `timeline.py`:per-unit 相位/上下文区间,可导出 JSONL(Gantt 数据)
- [x] `trace.py`:单测归因(REASONING / ACTING / QUEUED / WAITING)+ 可操作诊断
- [x] `demo trace` / `demo timeline` 子命令,支持从落盘日志离线分析
- [x] 交付闸门 `scripts/delivery_check.py`:测试 / lint / 双平台类型 / 确定性 /
      可观测性 / 性能 六项,当前 **6/6 PASS**

## R8 审查修复(已完成)

- [x] F1 fork 记账泄漏(高危)+ 回归测试
- [x] F2 reaper 误判进程死亡(高危)+ 4 个平台路径测试
- [x] F3/F4 圈复杂度 22/14 → 全部 ≤ 12
- [x] F5 死代码、F6 CWD 依赖、F7 异常分类
- [ ] F8 _sync_lease 轮询等待(技术债)
- [ ] F9 仿真 >12 agents 超线性(技术债)

## R5 扩展生态(已完成)

- [x] 入口点发现(`plugins.py`):backends / policies / tools 三组,懒加载
- [x] `PhaseWeaver(backend="name")` 按名解析;`build_policy` 插件回退
- [x] 失败信息含插件名与组名(`PluginError`);内置策略优先
- [x] 可运行第三方示例包 `examples/plugin_backend/`
- [x] 5 个单测(发现 / 损坏插件 / 未知名字 / 后端按名 / 策略插件)

- `[project.entry-points."phaseweaver.backends"]` 等入口点,第三方包可安装
- 提供一个"第三方风格"的示例包(`examples/plugin_backend/`),只依赖公开协议
- 判据:`pip install -e examples/plugin_backend` 后 `PhaseWeaver(backend=...)` 可用,
  且不 import 任何内部模块

## R6 调度与策略(已完成,负结果)

- [x] `AlignedPolicy`(流槽位准入)+ W11 异构工作负载 + 3 seeds 对照实验
- [x] 判决:**负结果,不采用**。异构场景 B2 goodput 0.1065 vs align2 0.0215(且 6/24 任务
      超时失败);均匀场景 align2 直接失败 10 个。
- [x] 结构性解释已记录(本地实验记录,不入库):占空比上界 `c̄ ≤ k·r/(r+t)` 由各 Agent
      自身 r/t 决定,对齐只改变瞬时重叠,不改变上界;准入闸门在容量已够时只会制造排队。
- [x] 副产品:发现并修复 `Scheduler.admit()` 是**从未被调用的死协议**(F11),现在它是真实的
      背压闸门(等待 + 超时),可被任何策略使用。

判据执行结果:未达到"goodput +5% 且 p99 不劣化",按约定记录负结果并停止,不调参凑数。

## R7 真实引擎验证(依赖外部条件)

- issue #3:真实 vLLM 上跑 L3;适配器已就绪,运行时不需要改动
- 判据:一张 `[L3]` 表 + 一次能力降级记录

## 不做的

- 不做多机/分布式调度(超出单机范围)
- 不做 Agent 编排 DSL(与"运行时"定位冲突)
- 不为了让数字好看去调参;负结果照写
