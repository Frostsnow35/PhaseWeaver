# 相关工作与能力矩阵

> 核实状态列区分"已阅读原文/文档"与"仅见标题或二手引用"。写作与引用时不得越过核实状态。

## 1. 服务层:让 Agent 的 KV Cache 在工具等待期让出 GPU

### Tokencake(已读摘要,arXiv:2510.18586)

> "multi-agent applications … space contention leads to the eviction of critical agents' caches and time underutilization leaves the cache of agents stalled on long-running tool calls idling in GPU memory."
> Space Scheduler 做动态内存分区以保护关键 Agent;Time Scheduler 在 function call 停顿期做**主动 offload + 预测性 upload**。相对 vLLM 端到端延迟降低 47.06%,有效显存利用率最高提升 16.9%。

来源:<https://arxiv.org/abs/2510.18586>

**与本项目的关系**:问题的表述几乎一致,是最直接的邻居与最强 baseline。差别在于它改造**引擎侧调度器与内存管理器**;PhaseWeaver 在**引擎之上**以 Agent 执行生命周期为控制面,通过能力协商的适配层表达同一类决策。因此本项目不能以"让 tool 等待期的 KV 让出 GPU"作为创新点,只能以"控制面位置、子 Agent 资源继承语义、引擎无关可迁移性、可恢复执行"作为差异化。

### Continuum:KV Cache TTL 驱动的多轮 Agent 调度(见标题与摘要片段)

给多轮 Agent 的 KV 缓存引入 time-to-live,在"重算成本"与"显存占用"之间做显式权衡。

来源:<https://www2.eecs.berkeley.edu/Pubs/TechRpts/2026/EECS-2026-234.html>

**与本项目的关系**:直接对应本规格的 B1 对照组(`TtlEvictPolicy`)。我们的 `retain_max_s` 与其 TTL 是同类机制,因此对照实验必须包含 B1,否则 B2 的收益无法归因。

### TAO:tool-aware 联合调度(仅见标题,待核对)

标题显示其主张为"针对 LLM Agentic Serving 的工具感知联合调度"。**引用前必须核实原文与实验设置**,当前不进入定位论证。

## 2. 推理引擎侧可编程能力(vLLM)

| 能力 | 粒度 | 对本项目的意义 | 来源 |
|---|---|---|---|
| `sleep(level=1\|2)` / `wake_up` | 引擎级 | 权重卸载 + **丢弃** KV;全局副作用,默认关闭,需显式开关 | <https://docs.vllm.ai/en/latest/features/sleep_mode/> |
| `abort` 请求 | 请求级 | 释放该请求 blocks;之后只能重算 → `DROP` 策略的实现基础 | 规格 §8 |
| `kv_events`(BlockStored/BlockRemoved) | 块级 | runtime 感知 KV 驻留的**真实**通道;mock 复用同一词汇表 | <https://docs.vllm.ai/en/latest/examples/features/kv_events/> |
| `reset_prefix_cache()` | 引擎级 | 前缀缓存整体失效;实验中用于隔离"前缀复用"干扰 | — |
| KV Connector / LMCache | 请求级(需部署) | `OFFLOAD` 策略的实现基础;未配置时降级为 `DROP` | <https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/> |

**结论**:PhaseWeaver 的"可编程资源协同"必须建立在上表的真实能力之上。任何声称"运行时把某个请求的 KV 搬到 CPU 再搬回"的设计,在不配置 KV connector 时都是不成立的。

## 3. 执行生命周期 / 持久化执行

- Durable execution 的经典做法(事件溯源 + 检查点 + 幂等回执)在 Temporal / Restate 一类系统中已是成熟范式;本项目第二阶段的差异是:检查点不仅覆盖业务状态,还覆盖**推理资源句柄**与**工具副作用回执**(ADR-0003)。
- 多 Agent 编排框架(LangGraph / OpenAI Agents SDK / AutoGen)提供图与状态,但不管理工具进程的取消与回收,也不与推理引擎的资源生命周期对话——这是 PhaseWeaver 的落点(ADR-0004)。

## 4. 差异化小结(写作时必须与上面对齐)

1. 控制面在 **Agent 执行生命周期**(工具进程的启动/取消/回收/状态交接),不是引擎内部调度器;
2. **Subagent 树**的资源继承与汇合语义(`SHARE` / `ISOLATE`,join 时记账合并、租约统一释放);
3. **引擎无关**:能力协商 + 四种处置 + 成本模型 + 静默降级,一套语义覆盖 vLLM / SGLang / 任意兼容端点;
4. **可恢复**(第二阶段):事件日志为基座的检查点与工具幂等回执。
