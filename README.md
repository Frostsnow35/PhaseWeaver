# PhaseWeaver

**Agent 执行运行时**:让 Agent 在"推理 ↔ 工具 ↔ 等待"之间的阶段切换,变成可编程的 GPU 资源决策。

[![CI](https://github.com/Frostsnow35/PhaseWeaver/actions/workflows/ci.yml/badge.svg)](https://github.com/Frostsnow35/PhaseWeaver/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 它解决什么

一个 Agent 任务通常是「推理 3 秒 → 跑工具 30 秒 → 再推理」。工具阶段 GPU 空转,但该 Agent 的
KV Cache 仍占着显存,其他 Agent 只能排队。PhaseWeaver 把这个切换点变成主动决策:
**保留 / 卸载 / 丢弃 / 释放**,并把代价写进账里;子 Agent 之间还能**移交 KV 所有权**。

> 它不是编排框架。契约只有一条:推理与工具调用经过 `unit.reason()` / `unit.call_tool()`,
> 让运行时看见阶段切换。详见 [它是什么](docs/guide/what-is-it.md)。

**Live demo**: <https://frostsnow35.github.io/PhaseWeaver/>

## 五分钟开始

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
- 项目展示页:[site/index.html](site/index.html)(单文件,浏览器直接打开,无外部依赖)

## 场景档案

| profile | 适用 | 默认策略 | 默认 KV 继承 |
|---|---|---|---|
| `coding` | 编码 Agent,短工具、长会话 | auto(压力自适应) | 不继承 |
| `research` | 浏览/检索,长工具 | B2(全局 SLO) | 不继承 |
| `orchestrator` | 多 Agent 扇出/汇合 | B2 | handoff |
| `delegation` | 链式委托(父→子→孙) | B2 | handoff |

一条命令跑完整场景矩阵:

```bash
python -m phaseweaver.demo scenarios --capacity-gib 2 --seeds 3
```

## 核心机制

- **双状态机**:`Phase`(执行阶段)与 `ContextState`(KV 归属)正交。
- **处置权租约**:`RETAIN / OFFLOAD / DROP / RELEASE` + 引擎能力协商 + 显式降级。
- **KV 交接**:fork 时把 KV 所有权交给子单元,join 交回——零拷贝、零重算。
- **可证下界**:按占空比推导的 makespan 下界,用来判断调度器离结构上限有多远。
- **事件日志记录**:日志整理记录各指标,不维护并行计数器。

## 接入真实引擎

```python
from phaseweaver.backends.vllm import VLLMBackend
from phaseweaver.backends.openai_compat import OpenAICompatConfig

backend = VLLMBackend(OpenAICompatConfig(base_url="http://127.0.0.1:8000", model="qwen2.5-7b"))
async with PhaseWeaver(backend=backend, profile="research") as pw:
    await pw.run("task", agent)
```

引擎不支持的能力会自动降级(如无 KV connector 时 `OFFLOAD → DROP`),事件里记录 `degraded_from`。
适配器协议见 [扩展指南](docs/guide/extending.md)。

## 开发

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src/phaseweaver
# 跨平台类型检查(CI 在两个 OS 上各跑一次)
uv run mypy --platform linux src/phaseweaver
```

工程约定见 [CONTRIBUTING.md](CONTRIBUTING.md),取舍记录见 [docs/adr/](docs/adr/),
开发历史与教训见 [docs/notes/devlog.md](docs/notes/devlog.md)。

### Demo 站点

静态页在 [`site/index.html`](site/index.html),由 GitHub Actions 发布到 GitHub Pages:

- 工作流:`.github/workflows/pages.yml`(构建 `site/` 制品 → `deploy-pages`)
- 仓库 Settings → Pages → **Source: GitHub Actions**(只需设置一次)
- 本地预览:`python -m http.server 8000 -d site`,或直接打开该文件
- 单文件、无外部请求、可离线打开;动效在 `prefers-reduced-motion` 下自动关闭

## 状态与边界

- 核心机制与实验均在 mock 仿真上完成([L1]/[L2]),**尚未在真实 GPU 上验证**([L3],issue #3)。
- 本仓库的性能数据依然为仿真数字验证的结果。
- 实验记录:[docs/notes/experiments.md](docs/notes/experiments.md)

## 本机环境注意事项(sandbox)

1. `uv` 缓存需重定向:`$env:UV_CACHE_DIR=<repo>\.uv-cache`。
2. 不使用 pytest 的 `tmp_path`:`%TEMP%` 与 `\\?\` 长路径在本机 sandbox 下不可访问。
3. 子进程不能用管道捕获 stdio(会 `WinError 5`),工具执行走文件重定向。
4. `taskkill` 在当前 token 下被拒绝,进程树回收改用 **Windows Job Object**。
5. `tempfile.mkdtemp`(mode 0700)创建的目录不可写,私有临时目录用默认权限自建。
6. `git push` 直接跑会失败(凭证管理器与 schannel 都被挡):用 `pwsh scripts/push.ps1`。

## 许可证

MIT
