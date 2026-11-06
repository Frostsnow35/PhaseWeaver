# PhaseWeaver

面向 Agent 执行生命周期与 GPU 推理资源协同的智能运行时。

PhaseWeaver 位于 Agent 编排框架与 LLM 推理引擎之间,把 Agent、Subagent 与工具调用统一抽象为可管理的执行单元,在 Agent 于「模型推理 / 工具执行 / 等待」之间切换时,通过**资源租约**与**处置策略**协调推理侧 KV Cache 的保留、释放、卸载与恢复。

- 设计规格:[docs/specs/2026-11-07-phaseweaver-design.md](docs/specs/2026-11-07-phaseweaver-design.md)
- 关键取舍(ADR):[docs/adr/](docs/adr/)
- 相关工作与能力矩阵:[docs/notes/related-work.md](docs/notes/related-work.md)

## 当前状态

第一阶段开发中。里程碑与完成判据见规格 §14。

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
