# PhaseWeaver

English | [中文](https://github.com/Frostsnow35/PhaseWeaver/blob/main/README_cn.md)

An agent execution runtime that turns phase switches between **reasoning ↔ tools ↔ waiting**
into programmable GPU resource decisions.

Documentation: <https://frostsnow35.github.io/PhaseWeaver/>

<a id="problem"></a>
## What it solves

An agent task is usually "reason for 3 s → run a tool for 30 s → reason again". During the tool
phase the GPU is idle, but that agent's KV Cache still holds memory and other agents have to queue.
PhaseWeaver turns this switch point into an active decision — **retain / offload / drop / release** —
and books the cost. Child agents can also **transfer KV ownership** between each other.

<a id="start"></a>
## Five-minute start

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

No GPU required: the default is a deterministic simulation backend (virtual clock + block-level
KV accounting) that reproduces exactly for the same seed.

- [Quickstart](docs/guide/quickstart.md) · [Extending](docs/guide/extending.md) · [Design spec](docs/specs/2026-11-07-phaseweaver-design.md)
- Project page: [site/index.html](site/index.html)

<a id="profiles"></a>
## Scenario profiles

| profile | use case | default policy | default KV inheritance |
|---|---|---|---|
| `coding` | coding agents, short tools, long sessions | auto (pressure-adaptive) | none |
| `research` | browsing / retrieval, long tools | B2 (global SLO) | none |
| `orchestrator` | multi-agent fan-out / join | B2 | handoff |
| `delegation` | chained delegation (parent → child → grandchild) | B2 | handoff |

After `pip install -e .`, the CLI works without a GPU:

```bash
phaseweaver scenarios --capacity-gib 2 --seeds 3
phaseweaver replay --trace examples/traces/sample.jsonl --seeds 0,1,2
```

`replay` compares pin-always against the global SLO scheduler on **your own** agent
trace (JSONL `reason` / `tool` events). The shipped trace is synthetic; the schema
is documented in `phaseweaver.replay`. All output is `[L1]` simulation.

<a id="mechanism"></a>
## Core mechanisms

- **Two orthogonal state machines**: `Phase` (execution stage) and `ContextState` (KV ownership).
- **Disposition leases**: `RETAIN / OFFLOAD / DROP / RELEASE` + engine capability negotiation + explicit degradation.
- **KV handoff**: fork transfers KV ownership to the child, join transfers it back — zero copy, zero recompute.
- **Provable lower bound**: a makespan bound derived from duty cycle, used to measure how far the scheduler is from the structural ceiling.
- **Event journal**: metrics are replayed from the log; no parallel mutable counters are maintained.

<a id="real-engine"></a>
## Connecting a real engine

```python
from phaseweaver.backends.vllm import VLLMBackend
from phaseweaver.backends.openai_compat import OpenAICompatConfig

backend = VLLMBackend(OpenAICompatConfig(base_url="http://127.0.0.1:8000", model="qwen2.5-7b"))
async with PhaseWeaver(backend=backend, profile="research") as pw:
    await pw.run("task", agent)
```

Capabilities the engine does not support degrade automatically (for example `OFFLOAD → DROP` when
no KV connector is available), with `degraded_from` recorded on the event.
See the [Extending guide](docs/guide/extending.md) for the adapter protocol.

<a id="dev"></a>
## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv run mypy src/phaseweaver
# cross-platform type checking (CI runs once on each OS)
uv run mypy --platform linux src/phaseweaver
```

Engineering conventions: [CONTRIBUTING.md](CONTRIBUTING.md); PR rules: [AGENTS.md](AGENTS.md).
Experiments, dev logs and decision records stay local (`_local/`, not committed) — the repository
only holds reproducible scripts and contract documents.

### Demo site

The static pages live at [`site/`](site/) and are published by GitHub Actions:

- Workflow: `.github/workflows/pages.yml` (builds the `site/` artifact → `deploy-pages`)
- Repository Settings → Pages → **Source: GitHub Actions** (one-time setup)
- Local preview: `python -m http.server 8000 -d site`, or just open a file
- Single file, no external requests, works offline; animations respect `prefers-reduced-motion`

<a id="status"></a>
## Status and limits

- Core mechanisms and experiments run on a deterministic simulator (`[L1]`/`[L2]`); **no end-to-end validation on a real inference engine yet** (issue #3).
- A `[L3] micro-benchmark` calibration on Kaggle T4 x2 (2026-11-10) replaced guessed constants: `bytes_per_token=57344` confirmed exactly, decode 33–34 tok/s vs mock 45, PCIe 12.22 GB/s vs 12.88, `contention_alpha≈0.018` vs 0.15. Parameter calibration, not a performance result.
- Real prefill time is superlinear in prompt length; the simulator's constant `prefill_tok_per_s` is a known model limitation (`scripts/probe_calibration.py`).
- Reproduce: `python scripts/run_experiments.py`, `probe_prefix.py`, `probe_capacity.py`, `probe_calibration.py` (raw rows stay in `runs/`, not committed).

<a id="sandbox"></a>
## Local sandbox notes (Windows)

1. Redirect the `uv` cache: `$env:UV_CACHE_DIR=<repo>\.uv-cache`.
2. Do not use pytest's `tmp_path`: `%TEMP%` and `\\?\` long paths are inaccessible in this sandbox.
3. Subprocess stdio cannot be captured through pipes (`WinError 5`); tools use file redirection.
4. `taskkill` is denied under the current token; process-tree reaping uses a **Windows Job Object**.
5. Directories created by `tempfile.mkdtemp` (mode 0700) are not writable; create private temp dirs with default permissions.
6. `git push` works with `http.sslBackend=schannel` and `credential.helper=manager`;
   the sandbox still denies writing the credential store (`Permission denied`). If credentials
   are not cached, use `pwsh scripts/push.ps1` to inject a `gh auth token` per invocation.

<a id="license"></a>
## License

MIT

[![CI](https://github.com/Frostsnow35/PhaseWeaver/actions/workflows/ci.yml/badge.svg)](https://github.com/Frostsnow35/PhaseWeaver/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
