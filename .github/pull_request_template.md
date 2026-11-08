## 改了什么

<!-- 一句话说清;关联 issue:Closes #123 -->

## 为什么

<!-- 问题背景;若推翻既有取舍,在描述里说明理由与备选方案(取舍记录不入库,见 AGENTS.md) -->

## 复现命令

```bash
# 干净环境可复制的完整命令(含容量、seed、策略/工作负载)
```

## 结果(必填,L1/L2/L3 标注)

| 指标 | 改动前 | 改动后 | 证据等级 |
|---|---|---|---|
| | | | |

- 配置:容量 ___ GiB;seeds ___;SLO 口径 ___
- 负结果(若有):<!-- 不采用的实验也要写在这里,并说明原因 -->

> 禁止把 L1/L2(mock/仿真)数据表述为真实 GPU 结果。

## 影响面

<!-- 是否改变不变量(I1–I8)、事件类型、对外契约、扩展点? -->

## 风险与回滚

<!-- 最坏情况;回滚方式(通常 revert 本 PR) -->

## 自查(对照 AGENTS.md)

- [ ] `uv run pytest -q` 全绿
- [ ] `uv run ruff check .` 通过
- [ ] `uv run mypy --platform linux src/phaseweaver` 通过
- [ ] `uv run mypy --platform win32 src/phaseweaver` 通过
- [ ] `python scripts/delivery_check.py --quick` 6/6
- [ ] 未提交实验/开发/取舍记录、运行产物或缓存(`git status --short` 自查)
- [ ] 契约变化已更新 `docs/specs/`,扩展点变化已更新 `docs/guide/`
