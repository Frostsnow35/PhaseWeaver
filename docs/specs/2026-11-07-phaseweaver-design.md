# PhaseWeaver 设计规格

- 状态:已批准(v1)
- 日期:2026-11-07
- 范围:第一阶段(Phase 1)——可运行的垂直切片 + 可迁移的引擎适配层
- 目标读者:实现本项目的编码 Agent 与维护者

---

## 1. 问题陈述

Agent 负载的执行生命周期在结构上与传统的 LLM 推理请求不同:一个任务由"模型推理 → 外部工具执行(秒级到分钟级)→ 模型推理"交替组成。在工具执行期间,Agent 不占用 GPU 算力,但它在推理引擎中已分配的 KV Cache 通常仍驻留在显存里(或只是被动地被引擎的调度器按时间配额驱逐),既无法被其他 Agent 使用,又无法被主动管理。

现有工作(如 Tokencake、Continuum、TAO)主要在**推理服务层**解决这个问题:改造引擎的调度器和内存管理器,让它在工具等待期主动 offload/驱逐。

PhaseWeaver 选择另一个切口:**把 Agent 的执行生命周期本身作为控制面**。运行时位于 Agent 编排框架与推理引擎之间,统一管理 Agent / Subagent / 工具调用的执行状态,并在阶段转换的瞬间,以"租约 + 处置策略"的方式协调推理引擎侧的 KV Cache 生命周期。它不重新实现编排框架,也不重新实现推理引擎。

## 2. 目标与非目标

### 2.1 目标(第一阶段)

- G1 统一的执行单元模型:Agent、Subagent、工具调用都是可管理的执行单元,支持树形 fork/join 与树形取消。
- G2 显式的阶段状态机:阶段转换是唯一的事实来源,所有转换落事件日志。
- G3 推理上下文与执行阶段解耦:非 GPU 阶段不得持有活跃的 KV 上下文(可断言的不变量)。
- G4 资源租约与预算准入:KV 容量以字节为单位做账,超额时按策略抢占。
- G5 引擎无关的适配层:通过能力协商支持 `RETAIN / DROP / OFFLOAD / RELEASE` 四种处置;mock 后端可完整仿真,真实后端不可用时自动降级。
- G6 工具执行真做:Python 与 Shell 执行器具备超时、取消、进程树回收能力;Browser 只留接口。
- G7 指标可从事件日志重放得到;能在无 GPU 的开发机上产出机制对比数据(标注可信度).
- G8 为第二阶段(Durable Runtime)预留接缝,但不实现检查点与恢复。

### 2.2 非目标(明确排除)

- 不实现或修改推理引擎内核(不写 CUDA、不 fork vLLM)。
- 不做分布式/多机调度、不做 K8s 集成。
- 不做 RL/学习型调度器;调度策略保持简单、可解释、可消融。
- 不实现真实浏览器自动化(Playwright 不引入)。
- 不做 Web UI;可视化输出为终端表格 + 可选 matplotlib 图。
- 不做多租户鉴权与计费。

## 3. 术语

| 术语 | 含义 |
|---|---|
| ExecutionUnit | 执行单元。树形结构,节点为 agent / subagent / tool call |
| Phase | 执行阶段,8 个取值,见 §5.1 |
| InferenceContext | 推理上下文,即一次连续性推理的 KV 归属;与 Phase 正交 |
| Lease | 对某类资源(字节/槽位)的占用声明,带优先级与过期时间 |
| Disposition | 处置策略:`RETAIN`/`DROP`/`OFFLOAD`/`RELEASE` |
| BackendCaps | 适配器能力声明,决定哪些处置可用 |
| Park | 使上下文退出 ACTIVE,让出显存但保留可恢复性 |
| Restore | 使上下文重新 ACTIVE,可能伴随重算或搬运 |
| Journal | 追加写事件日志,系统的唯一事实来源 |

## 4. 架构

```
        用户 Agent 代码 / 示例任务
                  │  async API
┌─────────────────▼──────────────────────────────────────────┐
│ Runtime (进程内, asyncio)                                   │
│  ├ UnitRegistry      单元树、fork/join、树形取消            │
│  ├ PhaseMachine      阶段状态机 + 转移表 + 不变量校验       │
│  ├ LeaseBroker       预算准入、抢占、过期回收               │
│  ├ PolicyEngine      YieldOnTool / RetainAlways / TTL       │
│  ├ ToolSupervisor    子进程监管、超时、进程树回收           │
│  ├ Journal           append-only JSONL                      │
│  └ MetricsReducer    从 journal 重放得出 RunReport          │
└─────────────────┬──────────────────────────────────────────┘
                  │ InferenceBackend 协议
    ┌─────────────┼──────────────┬────────────────┐
 MockBackend   VLLMBackend   OpenAICompatBackend  (未来: SGLang)
 离散事件仿真   OpenAI兼容+   任意兼容端点
               admin/kv_events (仅 DROP)
```

依赖方向严格单向:上层不 import 下层实现,`backends/*` 不 import `runtime`,只实现 `protocols.py` 中的协议。`policy` 只看到只读快照(`ClusterSnapshot`),不能直接改动状态。

## 5. 执行阶段与推理上下文

### 5.1 阶段(Phase)

8 个取值,7 个非终态转换源 + 3 个终态:

| Phase | 语义 | GPU |
|---|---|---|
| `PENDING` | 已创建,等待准入 | 否 |
| `QUEUED` | 已准入,等待引擎槽位 | 否 |
| `REASONING` | 正在进行 prefill/decode | 是 |
| `ACTING` | 正在执行工具(Python/Shell/Browser) | 否 |
| `WAITING` | 等待子单元汇合 / 结果未归 / 退避重试 | 否 |
| `COMPLETED` | 正常结束(终态) | 否 |
| `FAILED` | 失败结束(终态) | 否 |
| `CANCELLED` | 被取消(终态) | 否 |

### 5.2 合法转移表(唯一权威定义)

| From | To | 触发 |
|---|---|---|
| PENDING | QUEUED | `admit` 通过准入 |
| PENDING | CANCELLED | 父取消 / 超时 |
| QUEUED | REASONING | 引擎槽位与 KV 租约就绪 |
| QUEUED | CANCELLED | 父取消 / 超时 |
| REASONING | ACTING | 模型输出 tool call,开始执行工具 |
| REASONING | WAITING | 等待子 Agent 汇合 |
| REASONING | COMPLETED | 模型输出最终答案 |
| REASONING | FAILED | 引擎错误 / 重试耗尽 |
| REASONING | CANCELLED | 取消 |
| ACTING | REASONING | 工具结果返回(无论 ok/error,由错误策略决定后续) |
| ACTING | FAILED | 工具错误且 `on_tool_error=abort` |
| ACTING | CANCELLED | 取消 |
| WAITING | REASONING | 汇合满足 / 重试唤醒 |
| WAITING | FAILED | 等待超时 |
| WAITING | CANCELLED | 取消 |

任何不在表中的转移必须抛出 `IllegalTransition`,并在抛出前写入一条 `transition_rejected` 事件(便于诊断)。

### 5.3 推理上下文状态

| State | 语义 | 占租约 |
|---|---|---|
| `NONE` | 从未开始推理 | 否 |
| `ACTIVE` | 驻留 GPU,参与调度 | 是(kv_bytes) |
| `RETAINED` | 留在 GPU,不参与调度,可被抢占 | 是(kv_bytes) |
| `OFFLOADED` | 已移出 GPU(KV connector 或等价机制) | 否(仅可持有 `OFFLOAD_BYTES` 租约,用于约束落到 CPU 的份量) |
| `DROPPED` | 已 abort,仅保留重建所需元数据 | 否 |
| `RELEASED` | 终态,不可恢复 | 否 |

合法上下文转移:
`NONE→ACTIVE`、`ACTIVE→{ACTIVE(续推), RETAINED, OFFLOADED, DROPPED, RELEASED}`、
`RETAINED→{ACTIVE, OFFLOADED, DROPPED, RELEASED}`、`OFFLOADED→{ACTIVE, RELEASED}`、
`DROPPED→{ACTIVE(重算), RELEASED}`、`RELEASED` 无出边。

### 5.4 不变量(必须由 property test 强制)

- **I1 让出 GPU**:`phase ∈ {ACTING, WAITING}` ⟹ `context.state ≠ ACTIVE`
- **I2 租约一致**:`context.state ∈ {ACTIVE, RETAINED}` ⟺ 存在且仅存在一个匹配 `context_id` 的有效 `KV_BYTES` 租约
- **I3 预算不超发**:`Σ active_leases.amount ≤ budget.total_bytes`
- **I4 终态干净**:`phase ∈ {COMPLETED, FAILED, CANCELLED}` ⟹ 上下文为 `RELEASED` 且该单元无有效租约
- **I5 树不孤立**:父单元处于终态 ⟹ 所有后代单元均处于终态
- **I6 日志完备**:每次成功的阶段转移在 journal 中恰好出现一次,`seq` 严格单调递增
- **I7 取消有界**:取消请求在 `grace_ms + 500 ms` 内使子树全部到达终态,且不遗留存活子进程

不变量在每次状态变更后由 `InvariantChecker` 校验(`strict` 模式抛错并 dump journal 尾部,`audit` 模式仅记录违规事件)。测试默认 `strict`。

## 6. 单元树与 Subagent 语义

```python
class UnitKind(StrEnum):
    AGENT, SUBAGENT, TOOL_CALL
```

- `UnitRegistry.fork(parent, label, mode) -> AgentUnit`,`mode ∈ {SHARE, ISOLATE}`
  - `SHARE`:子单元从父的租约池借额度,预算共享,父被抢占时子随之降级;
  - `ISOLATE`:子单元从总预算中切出独立配额,父的抢占不影响子(配额不足则 fork 失败)。
- `join(child)`:把子单元的指标与用量并入父的 `aggregate`,释放子单元全部租约;子单元须已处于终态,否则抛 `JoinBeforeTermination`。
- `cancel(unit)`:深度优先后代优先;每个工具单元执行 `terminate → grace → kill`;等待收到 `TerminationReport` 后才返回;取消本身是幂等的。
- 单元 ID 使用单调递增的 `run_id:path` 形式(例如 `r1/root/sub-0/tool-3`),保证日志可读且可排序。

## 7. 租约与预算

```python
@dataclass(frozen=True)
class Lease:
    lease_id: str; unit_id: str; context_id: str | None
    kind: LeaseKind                # KV_BYTES | OFFLOAD_BYTES | ENGINE_SLOT | TOOL_SLOT
    amount: int                    # 字节或槽位数
    priority: int                  # 越小越优先
    expires_at: float | None
```

- `LeaseBroker.acquire(request) -> Lease | Rejected(cause)`;准入条件:`Σ + amount ≤ total - reserved_headroom`。
- 超额时执行**抢占**:候选**只能是 `RETAINED` 上下文**——由 I1 可知,`ACTIVE` 上下文的单元必然处于 `REASONING`,抢占它等于打断正在进行的推理,那是引擎调度器的职责(由后端仿真 / 引擎自行决定),不属于 PhaseWeaver 的租约回收。候选按 `(priority, idle_since)` 排序(高优先级先保、空闲久者先让),依次要求策略降级 `RETAIN → OFFLOAD → DROP`;仍不足则拒绝入队(`admission_reject`)。
- 抢占动作由 `BackendAdapter.park()` 执行,broker 只做决策与记账,不触碰引擎。
- 租约过期(若设置)由 `LeaseReaper` 定时扫描回收,回收事件落日志。
- 所有租约在 `Runtime.shutdown()` 时强制释放,并在日志中标记 `forced_release`(用于检验 I4)。

## 8. 处置策略(Disposition)

| 策略 | 适配器动作 | 成本模型 | 选择条件 |
|---|---|---|---|
| `RETAIN` | 无操作,继续占租约 | 0 恢复成本,但占用容量 | 预期工具时长 < `retain_max_s`(默认 2 s) |
| `OFFLOAD` | 经 KV connector 移出 | `bytes / mem_bw` + 回传延迟 | 预期时长 ∈ [`retain_max_s`, `offload_max_s`)(默认上限 20 s)且适配器支持 |
| `DROP` | `abort` 请求,释放 blocks | `recompute_tokens = resident_tokens` | 预期时长 ≥ `offload_max_s`,或 `OFFLOAD` 不可用,或预算紧张 |
| `RELEASE` | 单元终态,彻底放弃 | 0 | 终态 |

`BackendCaps` 声明:`supports_drop`、`supports_offload`、`supports_retain`、`supports_engine_sleep`、`capacity_bytes`、`mem_bw_bps`、`block_tokens`、`bytes_per_token`。不支持的能力在决策时**静默降级**(`OFFLOAD→DROP`,`RETAIN→DROP`),降级原因落日志字段 `degraded_from`。

默认策略 `YieldOnToolPolicy`:
1. 进入 `ACTING` 前,用 `expected_tool_duration_s`(来自 `ToolSpec` 声明;未声明时取同类工具的历史 P50,再缺失则用 `default_expected_s = 5.0`)按 §8 表的三档分桶;
2. 按分桶结果选择处置(预算紧张时整体下移一档:`RETAIN → OFFLOAD → DROP`);
3. 离开 `ACTING` 时立即 `restore`,并记录 `restore_latency_ms` 与 `recompute_tokens`。

对照组策略:`PinAlwaysPolicy`(B0)、`TtlEvictPolicy`(B1)、`OraclePolicy`(B3,零成本立即恢复)。

## 9. 工具执行

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str; executor: Literal["python", "shell", "browser"]
    timeout_s: float; grace_ms: int = 2000
    expected_duration_s: float | None = None
    idempotency_key: str | None = None      # 第二阶段使用
```

- 统一结果信封:`ToolResult(status: ok|error|timeout|cancelled|killed, stdout, stderr, exit_code, duration_ms, truncated)`。
- `PythonExecutor`:在临时工作目录内以 `python -I -c` 运行一段包装脚本,stdin 传 JSON 参数,stdout 返回单行 JSON;`-I` 隔离用户 site-packages 影响。
- `ShellExecutor`:`create_subprocess_shell`,平台化进程组。
- `BrowserExecutor`:实现 `execute()` 但立即返回 `status="error", stderr="browser executor not implemented"`;接口与其余执行器完全一致,未来替换不需改 runtime。
- 输出上限:`max_stdout_bytes`(默认 1 MiB),超出截断并置 `truncated=True`。
- `ProcessReaper`:平台适配层
  - POSIX:`start_new_session=True`,取消时 `killpg(SIGTERM)` → `grace_ms` → `killpg(SIGKILL)`;
  - Windows:`CREATE_NEW_PROCESS_GROUP`,取消时 `taskkill /T /F /PID`,无 `psutil` 时退化为 `process.terminate()` + 枚举子进程;
  - 两者都必须在取消后确认进程树消失,否则记录 `reap_incomplete` 事件(I7 依赖此点)。

## 10. 事件日志

- 单文件 JSONL + 内存索引;`seq` 从 1 单调递增;每条含 `seq, ts, run_id, event, unit_id, phase_from, phase_to, cause_seq, payload`。
- 事件类型(封闭集合,新增需在本表登记):
  `unit_created`、`phase_transitioned`、`transition_rejected`、`context_state_changed`、
  `lease_requested`、`lease_granted`、`lease_released`、`lease_preempted`、`admission_rejected`、
  `park_started`、`park_completed`、`restore_started`、`restore_completed`、
  `tool_started`、`tool_finished`、`tool_killed`、`reap_incomplete`、
  `invariant_violated`、`run_started`、`run_finished`、`forced_release`。
- 所有指标由 `MetricsReducer.replay(journal) -> RunReport`,**不维护独立的可变计数器**。
- `RunReport`:per-task 延迟数组、`p50/p95`、`goodput`、`peak_resident_bytes`、`gpu_idle_wasted_bytes_s`、`recompute_tokens_wasted`、`park_counts`、`restore_latency_p95_ms`、`admission_rejects`、`jain_fairness`。
- 第二阶段接缝:journal 已带 `seq` 与因果链,检查点只需 `(last_seq, snapshot)`;工具幂等回执只需在 `tool_started/finished` 上挂 `idempotency_key` 与 `result_hash`。第一阶段不实现。

## 11. Mock 后端契约

`MockBackend` 是第一阶段唯一的数字来源,必须严格、确定、可解释。

- **虚拟时钟**:`VirtualClock`,`now()` 由仿真推进;所有等待以时钟推进实现,不调用 `time.sleep`。`--real-time` 模式下以真实时间映射虚拟时间,仅用于演示。
- **参数**(`MockProfile`):`block_tokens=16`、`bytes_per_token`(默认 57344,对应 Qwen2.5-7B bf16 的每 token KV)、`capacity_bytes=8 GiB`、`prefill_tok_per_s=3200`、`decode_tok_per_s=45`、`mem_bw_bps=12 GiB/s`、`queue_limit`。
- **资源模型**:KV 以 block 为粒度分配;分配失败触发仿真内抢占(与 vLLM 的 preemption 语义一致),抢占产生 `preempted` 引擎事件。
- **引擎事件**:`BlockStored`、`BlockRemoved`、`RequestPreempted`、`QueueDepthChanged`,与 vLLM `kv_events` 使用同一套词汇,runtime 走完全相同的感知路径(禁止任何"仅 mock 可用"的旁路)。
- **确定性**:相同 `seed` 与相同工作负载 ⇒ 逐事件一致的 journal(测试用 `journal_digest` 断言)。
- **`gpu_idle_wasted_bytes_s` 定义**:对所有 `t`,若 `active_inference_count(t) == 0`,则该时段内驻留的 KV 字节数 × 时长累加。这是"让出 GPU"收益的量化单位。

## 12. 指标与实验设计

### 12.1 对照与消融

| 代号 | 配置 |
|---|---|
| B0 | `PinAlwaysPolicy`:KV 一律保留到单元结束,无让出、无抢占(按容量 FIFO 排队)—— 主流 Agent 框架 + 引擎默认行为的近似 |
| B1 | `TtlEvictPolicy`(固定 TTL 释放,Continuum 式思路) |
| B2 | `YieldOnToolPolicy`(PhaseWeaver) |
| B3 | `OraclePolicy`(进入 ACTING 即零成本 DROP、返回瞬时恢复)——理论上界 |

消融维度:预算(`2/4/8/16 GiB`)、`retain_max_s`、工具时长分布(全短 / 全慢 / 混合)、能力降级(禁用 `OFFLOAD`)、子 Agent 模式(`SHARE` vs `ISOLATE`)。

### 12.2 工作负载

- W1 同质:8 个 Agent,各 3 轮,工具时长 lognormal(P50=2s)。
- W2 混合长短:4 个"浏览器型"长工具(30–60s)+ 6 个短工具,压测容量争抢。
- W3 扇出:1 个父 Agent fork 4 个子 Agent 各 2 轮后 join,验证租约继承与汇合记账。

### 12.3 核心主张与判据

主张:**工具执行期间让出推理资源,能提升同墙钟时间内的完成量并降低排队延迟;代价是可控的重算。**

判据(B2 vs B0,在 8 GiB 预算、W2 下的期望方向):
`goodput` 上升、`p95 端到端延迟` 下降、`gpu_idle_wasted_bytes_s` 显著下降;
并且 `recompute_tokens_wasted` 的增长必须被 goodput 收益覆盖(报告里同时给出两者,不做选择性呈现)。

### 12.4 数字可信度分级(强制标注在每张表上)

- **L1 机制证明**:mock 仿真的对比结果。证明"机制按设计工作",不作为性能结论。
- **L2 参数敏感性**:同一仿真下的消融与扫描。
- **L3 真实测量**:`VLLMBackend` + 真实 GPU。**第一阶段不产出 L3**;规格要求所有文档不得把 L1/L2 表述为 L3。

## 13. 仓库结构

```
PhaseWeaver/
├─ pyproject.toml              # hatchling, requires-python >=3.11
├─ README.md
├─ docs/
│  ├─ specs/2026-11-07-phaseweaver-design.md
│  ├─ plans/                   # 实施计划(writing-plans 产出)
│  ├─ adr/0001..0004           # 关键取舍
│  └─ notes/related-work.md
├─ src/phaseweaver/
│  ├─ __init__.py  types.py  errors.py  protocols.py
│  ├─ phases.py  units.py  leases.py  policy.py
│  ├─ journal.py  metrics.py  invariant.py  clock.py
│  ├─ runtime.py
│  ├─ tools/{__init__,base,python_exec,shell_exec,browser_exec,reaper}.py
│  └─ backends/{__init__,base,mock,vllm,openai_compat}.py
├─ tests/{unit,property,chaos,integration}/
└─ examples/{demo_compare.py,workloads/*.yaml}
```

模块纪律:单文件目标 ≤ 400 行;超过即拆分。`phases.py` / `leases.py` / `invariant.py` 不允许 import `runtime.py`(依赖环由 ruff 的 `TID`/自定义规则或纯人工评审保证,以 import 图测试固化)。

## 14. 里程碑与完成判据

| 里程碑 | 内容 | 完成判据 |
|---|---|---|
| M0 | 骨架、`types`、`phases`、FSM、journal、pytest 环境 | 非法转移被拒的测试全绿;`journal` 追加与读取测试通过 |
| M1 | 单元树、fork/join、树形取消、Python+Shell 执行器、`ProcessReaper` | Windows 上取消长任务后无存活子进程(以进程树检查断言) |
| M2 | `LeaseBroker`、预算准入、四种处置、`YieldOnToolPolicy`、不变量 | hypothesis property test 不变量零违反;预算不超发 |
| M3 | `MockBackend` 全量仿真、`MetricsReducer`、`demo compare` | 本机 3 分钟内跑完 W1–W3 并输出 B0/B2/B3 对比表;`pytest` 全绿 |
| M4 | `VLLMBackend`、`OpenAICompatBackend`、README、ADR、related-work | 对 OpenAI 兼容端点可跑通一次真实调用(或明确记录未验证) |
| M5(独立 spec) | 检查点、事件重放恢复、工具幂等回执 | 不在本规格范围内 |

## 15. 技术选型

- Python ≥ 3.11(开发机 3.12.4),全程 `asyncio`;`uv` 管理依赖与虚拟环境。
- 运行时核心**零第三方依赖**;`psutil` 仅作为 Windows 进程回收的可选增强;`hypothesis`、`pytest`、`ruff`、`mypy`、`matplotlib` 为开发依赖。
- 类型:`mypy --strict` 覆盖 `src/phaseweaver` 的 core 模块(`types/phases/leases/invariant/journal/metrics`);backends 与 tools 放宽到 `--check-untyped-defs`。
- 风格:中文文档 + 英文代码与提交信息;提交前缀 `feat/fix/test/docs/chore`。
- 许可证 MIT。

## 16. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| mock 与真实引擎语义漂移 | 迁移时机制失效 | mock 复用 kv_events 词汇表;M4 增加 parity 冒烟用例;文档标注 L1/L2 |
| vLLM 无请求级 offload | `OFFLOAD` 在生产不可用 | 能力协商降级为 `DROP`;显式记录 `degraded_from` |
| Windows 进程回收语义弱 | 取消不干净,I7 失效 | 独立 `ProcessReaper` + 平台化测试 + `reap_incomplete` 事件告警 |
| 范围膨胀(调度算法、分布式、UI) | 第一阶段做不完 | §2.2 非目标 + 里程碑判据卡口 |
| 数字被误读为性能结论 | 作品可信度受损 | §12.4 强制分级标注,表头必须出现 L1/L2 |
| 引擎级 sleep/wake 影响同租户 | 真实环境副作用 | 默认关闭,需显式 `--allow-engine-sleep`,并在报告中单列 |

## 17. 已解决的设计问题(记录取舍,避免反复)

1. 阶段与上下文正交(而非合并状态机)——否则 `ACTING` 同时"要释放 GPU"与"随时恢复"不可表达。见 ADR-0001。
2. 租约抽象为"处置权 + 成本模型",而非"借出内存"——引擎不提供可搬动的内存句柄。见 ADR-0002。
3. 第一阶段即引入事件日志——指标可重放,第二阶段的持久化直接复用。见 ADR-0003。
4. 进程内 asyncio 运行时而非 sidecar——先验证机制,留 transport 缝。见 ADR-0004。
5. Browser 只做接口——价值集中在生命周期与资源协同,不在浏览器自动化。
