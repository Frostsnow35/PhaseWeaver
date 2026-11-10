# PhaseWeaver

[English](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README.md) | 中文

**Agent 执行运行时**:让 Agent 在「推理 ↔ 工具 ↔ 等待」之间的阶段切换,变成可编程的 GPU 资源决策。

文档:<https://frostsnow35.github.io/PhaseWeaver/>

<a id="problem"></a>
## 它解决什么 · [English](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README.md#problem)

一个 Agent 任务通常是「推理 3 秒 → 跑工具 30 秒 → 再推理」。工具阶段 GPU 空转,但该 Agent 的
KV Cache 仍占着显存,其他 Agent 只能排队。PhaseWeaver 把这个切换点变成主动决策:
**保留 / 卸载 / 丢弃 / 释放**,并把代价写进账里;子 Agent 之间还能**移交 KV 所有权**。

<a id="start"></a>
## 五分钟开始 · [English](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README.md#start)

```python
import asyncio
from phaseweaver import PhaseWeaver, ToolSpec

async def agent(unit):
    await unit.reason(prompt_tokens=2048, out_tokens=128)
    await unit.call_tool(ToolSpec("pytest", "python", expected_duration_s=4.0),
                         {"code": "print('ok')"})
    await unit.reason(prompt_tokens=2176, out_tokens=128)

async def main():
    async with PhaseWeaver(profile="coding", capacity_gib=2.0) as pw:
        await pw.run("fix-bug", agent)
        print(pw.summary())

asyncio.run(main())
```

无需 GPU:默认使用确定性仿真后端(虚拟时钟 + 块级 KV 记账),同 seed 结果可复现。

- [快速开始](docs/guide/quickstart.md) · [扩展指南](docs/guide/extending.md) · [设计规格](docs/specs/2026-11-07-phaseweaver-design.md)
- 交互实验场:<https://frostsnow35.github.io/PhaseWeaver/playground.html>([源码](site/playground.html);单文件,可离线,无需 GPU)
- 项目展示页:[site/index.html](site/index.html)

<a id="profiles"></a>
## 场景档案 · [English](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README.md#profiles)

| profile | 适用 | 默认策略 | 默认 KV 继承 |
|---|---|---|---|
| `coding` | 编码 Agent,短工具、长会话 | auto(压力自适应) | 不继承 |
| `research` | 浏览/检索,长工具 | B2(全局 SLO) | 不继承 |
| `orchestrator` | 多 Agent 扇出/汇合 | B2 | handoff |
| `delegation` | 链式委托(父→子→孙) | B2 | handoff |

`pip install -e .` 之后,CLI 无需 GPU 即可运行:

```bash
phaseweaver scenarios --capacity-gib 2 --seeds 3
phaseweaver replay --trace examples/traces/sample.jsonl --seeds 0,1,2
```

`replay` 用**你自己的** Agent 轨迹(JSONL 的 `reason` / `tool` 事件)对比 pin-always
与全局 SLO 调度器。仓库自带的轨迹是合成样例,格式见 `phaseweaver.replay` 模块文档。
所有输出均为 `[L1]` 仿真。

<a id="mechanism"></a>
## 核心机制 · [English](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README.md#mechanism)

- **双状态机**:`Phase`(执行阶段)与 `ContextState`(KV 归属)正交。
- **处置权租约**:`RETAIN / OFFLOAD / DROP / RELEASE` + 引擎能力协商 + 显式降级。
- **KV 交接**:fork 时把 KV 所有权交给子单元,join 交回——零拷贝、零重算。
- **可证下界**:按占空比推导的 makespan 下界,用来判断调度器离结构上限有多远。
- **事件日志记录**:日志整理记录各指标,不维护并行计数器。

<a id="real-engine"></a>
## 接入真实引擎 · [English](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README.md#real-engine)

```python
from phaseweaver.backends.vllm import VLLMBackend
from phaseweaver.backends.openai_compat import OpenAICompatConfig

backend = VLLMBackend(OpenAICompatConfig(base_url="http://127.0.0.1:8000", model="qwen2.5-7b"))
async with PhaseWeaver(backend=backend, profile="research") as pw:
    await pw.run("task", agent)
```

引擎不支持的能力会自动降级(如无 KV connector 时 `OFFLOAD → DROP`),事件里记录 `degraded_from`。
适配器协议见 [扩展指南](docs/guide/extending.md)。

<a id="dev"></a>
## 开发 · [English](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README.md#dev)

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src/phaseweaver
# 跨平台类型检查(CI 在两个 OS 上各跑一次)
uv run mypy --platform linux src/phaseweaver
```

工程约定见 [CONTRIBUTING.md](CONTRIBUTING.md),PR 规范见 [AGENTS.md](AGENTS.md);实验、开发与取舍记录只保留在本地(`_local/`,不入库),仓库只保留可复现脚本与契约文档。

### Demo 站点

静态页在 [`site/`](site/),由 GitHub Actions 发布到 GitHub Pages:

- 工作流:`.github/workflows/pages.yml`(构建 `site/` 制品 → `deploy-pages`)
- 仓库 Settings → Pages → **Source: GitHub Actions**(只需设置一次)
- 本地预览:`python -m http.server 8000 -d site`,或直接打开页面文件
- 单文件、无外部请求、可离线打开;动效在 `prefers-reduced-motion` 下自动关闭

<a id="status"></a>
## 状态与边界 · [English](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README.md#status)

- 核心机制与实验都在确定性仿真上完成([L1]/[L2]),**尚未在真实推理引擎上做端到端验证**(issue #3)。
- 2026-11-10 在 Kaggle T4 x2 完成 `[L3] micro-benchmark` 参数标定:`bytes_per_token=57344` 精确对齐;decode 33–34 tok/s(mock 45);PCIe 12.22 GB/s(mock 12.88);`contention_alpha≈0.018`(mock 0.15)。这是参数标定,不是性能结论。
- 真机 prefill 耗时随 prompt 长度超线性,仿真器的常数 `prefill_tok_per_s` 是已知模型限制(`scripts/probe_calibration.py`)。
- 复现:`python scripts/run_experiments.py`、`probe_prefix.py`、`probe_capacity.py`、`probe_calibration.py`(原始行留在 `runs/`,不入库)。

<a id="sandbox"></a>
## 本机环境注意事项(sandbox) · [English](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README.md#sandbox)

1. `uv` 缓存需重定向:`$env:UV_CACHE_DIR=<repo>\.uv-cache`。
2. 不使用 pytest 的 `tmp_path`:`%TEMP%` 与 `\\?\` 长路径在本机 sandbox 下不可访问。
3. 子进程不能用管道捕获 stdio(会 `WinError 5`),工具执行走文件重定向。
4. `taskkill` 在当前 token 下被拒绝,进程树回收改用 **Windows Job Object**。
5. `tempfile.mkdtemp`(mode 0700)创建的目录不可写,私有临时目录用默认权限自建。
6. `http.sslBackend=schannel` + `credential.helper=manager` 下 `git push` 可成功;沙箱仍会拒绝写 credential store(`Permission denied`)。凭据未缓存时用 `pwsh scripts/push.ps1` 单次注入 `gh auth token`。

<a id="license"></a>
## 许可证 · [English](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README.md#license)

MIT

[![CI](https://github.com/Frostsnow35/PhaseWeaver/actions/workflows/ci.yml/badge.svg)](https://github.com/Frostsnow35/PhaseWeaver/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
