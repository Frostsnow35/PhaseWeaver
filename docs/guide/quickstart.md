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
from phaseweaver import PhaseWeaver, ToolSpec

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

## 用你自己的 Agent 轨迹试(无需 GPU)

`phaseweaver replay` 吃 JSONL 轨迹,每行一个 `reason` 或 `tool` 事件:

```json
{"ts":0.0,"unit_id":"a0","kind":"reason","prompt_tokens":4096,"out_tokens":128,"prefix_id":"sys-v1","prefix_tokens":3072}
{"ts":3.2,"unit_id":"a0","kind":"tool","name":"pytest","tool_s":5.0}
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `ts` | float | 事件时间(秒),用于排序与阅读,不参与仿真推进 |
| `unit_id` | str | 同一 Agent 的事件归为一个执行单元 |
| `kind` | `reason` / `tool` | 必填,其他值直接报错 |
| `prompt_tokens`, `out_tokens` | int | `reason` 事件的 prefill / decode 长度 |
| `tool_s` | float | `tool` 事件的实际耗时 |
| `name` | str | 工具名,默认 `tool` |
| `prefix_id`, `prefix_tokens` | str / int | 可选;共享前缀标识与长度,仅用于静态块级重合度测量 |

```bash
phaseweaver replay --trace examples/traces/sample.jsonl --seeds 0,1,2
```

`prefix_id` / `prefix_tokens` 缺失时前缀重合度报 `unknown`,不猜数。
输出强制带 `[L1]` 标头与 seeds / C / SLO 脚注。仓库自带的 `sample.jsonl` 是合成样例,
不是真实轨迹,不得当作真实负载的证据。

## 用真机标定值复核结论

Kaggle notebook(`_local/kaggle_l3_calibration.ipynb`,由 `scripts/make_kaggle_notebook.py` 生成)
产出的 `calibration.json` 放进 `runs/kaggle-l3/` 后:

```bash
python scripts/probe_calibration.py
```

该脚本用真机常数与上下界跑 6 个 `[L2]` 变体,重算前缀命中与容量放大两条闸门,
原始行写 `runs/calibration-probe.json`。标定口径是 `[L3] micro-benchmark / parameter calibration`,
不是端到端性能结果;速率来自 0.5B、`bytes_per_token` 来自 7B,不能合并成同一份 `BackendCaps`。
`prefill` 随 prompt 长度非线性,不要回填 `BackendCaps.prefill_tok_per_s`。
