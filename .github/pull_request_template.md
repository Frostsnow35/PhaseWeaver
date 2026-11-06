## 做了什么

<!-- 一句话说明;关联 issue:Closes #123 -->

## 为什么

<!-- 问题背景;如果推翻/修改了已有 ADR,请指出 -->

## 验证

- [ ] `uv run pytest -q` 全绿
- [ ] `uv run ruff check .` 通过
- [ ] `uv run mypy src/phaseweaver` 通过
- [ ] 若修改了实验链路,已更新 `docs/notes/experiments.md`

## 影响面

<!-- 是否改变不变量(I1–I8)、事件类型、租约语义、调度判据? -->

## 数字(L1/L2 必须标注)

| 场景 | 改动前 | 改动后 |
|---|---|---|
| | | |

> 禁止把 L1/L2(mock)数字表述为真实 GPU 测量。
