# 快速开始

## 安装

```bash
uv sync --extra dev
# 或
pip install -e .
```

## 五行接入(仿真后端,无需 GPU)

```python
import asyncio
from phaseweaver import PhaseWeaver

async def agent(unit):
    await unit.reason(prompt_tokens=2048, out_tokens=128)
    await unit.call_tool(ToolSpec("tests", "python", expected_duration_s=5.0),
                         {"code": "print('ok')"})
    await unit.reason(prompt_tokens=2200, out_tokens=128)

async def main():
    async with PhaseWeaver(profile="coding", capacity_gib=2.0) as pw:
        outcome = await pw.run("demo", agent)
        print(outcome.phase, pw.summary())

asyncio.run(main())
```

`PhaseWeaver` 会根据 profile 自动选择:调度策略、KV 继承模式、预算余量、SLO 默认值。

## 四个场景档案

| profile | 适用 | 默认策略 | 默认 KV 继承 |
|---|---|---|---|
| `coding` | 编码 Agent,短工具、长会话 | auto(压力自适应) | 不继承 |
| `research` | 浏览/检索,长工具 | B2(全局 SLO) | 不继承 |
| `orchestrator` | 多 Agent 扇出/汇合 | B2 | handoff |
| `delegation` | 链式委托(父→子→孙) | B2 | handoff |

一条命令看四种场景的对比:

```bash
python -m phaseweaver.demo scenarios --capacity-gib 2 --seeds 3
```

## 接入真实引擎

```python
from phaseweaver import PhaseWeaver
from phaseweaver.backends.vllm import VLLMBackend
from phaseweaver.backends.openai_compat import OpenAICompatConfig

backend = VLLMBackend(OpenAICompatConfig(base_url="http://127.0.0.1:8000", model="qwen2.5-7b"))
async with PhaseWeaver(backend=backend, profile="research") as pw:
    await pw.run("task", agent)
```

传入真实 backend 时,门面会自动切到真实时钟、关闭工具仿真。引擎能力不足时租约显式降级
(`OFFLOAD → DROP`),并在事件里记录 `degraded_from`。

## 子 Agent 与 KV 交接

```python
async def parent(unit):
    await unit.reason(prompt_tokens=4096, out_tokens=128)
    # 交接:父把 KV 所有权交给子,子零 prefill 起步;join 后交回
    await unit.fork("child", child_fn)          # profile=orchestrator 时默认 handoff
    await unit.reason(prompt_tokens=4370, out_tokens=128)
```

需要串行接力(每个子都热启动)时用 `inherit="relay"`;想显式关闭用 `inherit="none"`。

## 排障:这个任务为什么慢

先留下日志,再离线分析(零运行时开销):

```python
from pathlib import Path
async with PhaseWeaver(profile="coding", journal_path=Path("runs/run.jsonl")) as pw:
    await pw.run("task", agent)
```

```bash
# 单测追踪:相位耗时、park/restore、交接、诊断结论
python -m phaseweaver.demo trace --journal runs/run.jsonl --unit run/root-0-0

# 全部单元一览
python -m phaseweaver.demo trace --journal runs/run.jsonl --all

# 相位时间线(Gantt 数据)
python -m phaseweaver.demo timeline --journal runs/run.jsonl --out runs/timeline.jsonl
```

`trace` 会把耗时归因到 `REASONING` / `ACTING` / `QUEUED` / `WAITING`,并给出可操作结论,
例如"ACTING 占 78%:工具本身是瓶颈——这不是调度能解决的"。

## 看指标

```python
print(pw.report().summary_row())
# {'completed': 1.0, 'slo_rate': 1.0, 'goodput': ..., 'p99_ms': ..., 'kv_waste_mib_s': ...}
```

所有指标由事件日志重放得出,可复现(同 seed 两次结果一致)。
