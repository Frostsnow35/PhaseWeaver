# PhaseWeaver 是什么,不是什么

## 一句话

**Agent 执行运行时**:位于 Agent 编排层与 LLM 推理引擎之间,把"Agent 正在推理 / 正在跑工具 /
正在等待"变成可编程资源决策,并保证子 Agent 树与失败路径上的资源不泄漏。

## 与"框架 / 库 / 运行时"的关系

| 形态 | 控制反转 | 例子 | 我们的位置 |
|---|---|---|---|
| 库 | 否,你调用它 | requests | 我们提供库级 API(`Runtime`) |
| 框架 | 是,它调用你 | LangGraph、AutoGen | **不是**。我们不接管 Agent 如何思考 |
| 运行时 | 部分,它托管资源生命周期 | 容器运行时、WASM runtime | **是**。我们托管执行单元与 KV 资源 |

所以正确的表述是:

> PhaseWeaver 是一个**运行时 + 窄契约**,不是编排框架。
> 契约只有一条:Agent 的推理与工具调用必须经过 `unit.reason()` / `unit.call_tool()`,
> 让运行时看见阶段切换。遵守这条,它可以嵌进任何编排框架。

## 三层 API

```python
# L1:场景档案,五行接入
from phaseweaver import PhaseWeaver

async with PhaseWeaver(profile="coding", capacity_gib=2.0) as pw:
    await pw.run("task", agent_fn)
    print(pw.summary())

# L2:显式控制(backend / budget / policy 自己拼)
from phaseweaver import Runtime, RuntimeConfig, Budget

runtime = Runtime(backend=..., clock=..., budget=Budget(...), policy=...)

# L3:扩展点(实现协议即可)
InferenceBackend   # 接一个新引擎
Scheduler          # 换一种资源策略
ToolExecutor       # 加一种工具类型
Profile            # 定义一个新场景
```

## 它解决的具体问题

一个 Agent 任务通常是"推理 3 秒 → 跑工具 30 秒 → 再推理"。在工具阶段:

- GPU 在空转,但该 Agent 的 KV Cache 还占着显存;
- 其他 Agent 因为显存不足而排队;
- 现有框架要么看不到这个阶段切换,要么只能等引擎自己的调度器被动驱逐。

PhaseWeaver 把这个切换点变成**主动决策**:保留 / 卸载 / 丢弃 / 释放,并把代价写进账里。

## 适用 / 不适用

**适用**:单机、多 Agent、工具耗时占比高、显存是瓶颈、Agent 之间存在父子委托关系。

**不适用**:单请求低延迟场景(没有阶段切换可优化)、多机分布式调度、想要"开箱即用的
Agent 编排框架"的场景。

## 现在的成熟度

- 核心机制与实验均在 **mock 仿真**([L1]/[L2])上完成,尚未在真实 GPU 上验证([L3],见 issue #3)。
- 不要把本仓库的仿真数字表述为真实引擎性能。
