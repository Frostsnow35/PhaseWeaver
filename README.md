# PhaseWeaver

面向 Agent 执行生命周期与 GPU 推理资源协同的智能运行时。

PhaseWeaver 位于 Agent 编排框架与 LLM 推理引擎之间,把 Agent、Subagent 与工具调用统一抽象为可管理的执行单元,在 Agent 于「模型推理 / 工具执行 / 等待」之间切换时,通过**资源租约**与**处置策略**协调推理侧 KV Cache 的保留、释放、卸载与恢复。

- 设计规格:[docs/specs/2026-11-07-phaseweaver-design.md](docs/specs/2026-11-07-phaseweaver-design.md)
- 关键取舍(ADR):[docs/adr/](docs/adr/)
- 相关工作与能力矩阵:[docs/notes/related-work.md](docs/notes/related-work.md)

## 当前状态

- M0–M3 完成:执行单元树、双状态机、租约/处置、全局 SLO 调度、mock 仿真、E1 CLI。
- 测试:`50 passed`(unit / property / integration),`mypy --strict` 与 `ruff` 通过。
- 实验:E1 已跑通,但**核心主张尚未成立**,原因与修复计划见 [docs/notes/experiments.md](docs/notes/experiments.md)。
- 下一步:F1 租约 block 化 → F2 mock 算力竞争 → F3 背压重设计 → F4 主指标修正 → F5 全量重跑。

## 本机环境注意事项(sandbox)

1. `uv` 缓存需重定向:`$env:UV_CACHE_DIR=<repo>\.uv-cache`。
2. 不使用 pytest 的 `tmp_path`:`%TEMP%` 与 `\\?\` 长路径在本机 sandbox 下不可访问。
3. 子进程不能用管道捕获 stdio(会 `WinError 5`),工具执行走文件重定向。
4. `taskkill` 在当前 token 下被拒绝,进程树回收改用 **Windows Job Object**。
5. `tempfile.mkdtemp`(mode 0700)创建的目录不可写,私有临时目录用默认权限自建。

## 快速开始

```bash
uv sync
uv run pytest -q
uv run python -m phaseweaver.demo compare --workload w1
```

## 数字可信度

本项目所有对比数据都标注可信度等级:**L1 机制证明**(mock 仿真)、**L2 参数敏感性**、**L3 真实测量**(真实 GPU + vLLM)。第一阶段只产出 L1/L2,不得表述为 L3。见规格 §12.4。

## 许可证

MIT
