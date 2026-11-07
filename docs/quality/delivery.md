# 交付标准

一条命令: `python scripts/delivery_check.py`(约 2–3 分钟,`--quick` 跳过性能闸门)。

CI 保证"不坏",这份标准保证**达标**。每一条都必须可执行、可复现。

## 闸门

| # | 闸门 | 判据 | 当前 |
|---|---|---|---|
| 1 | 测试 | `pytest` 全绿,0 fail | 98 passed |
| 2 | 静态检查 | `ruff check .` 无告警 | PASS |
| 3 | 类型 | `mypy --platform linux|win32 src/phaseweaver` 双平台 0 错 | 40 文件 PASS |
| 4 | 确定性 | 同 seed 两次运行 `summary_row()` 完全一致 | PASS |
| 5 | 可观测性 | 能从日志构建时间线 + 单测追踪,回放 < 200 ms | 5 spans / 0.1 ms |
| 6 | 性能 | 排队场景下 B2 的 SLO ≥ B0,且 p99 ≤ B0 × 1.02 | B2 slo=1.000 p99=109.4s vs B0 0.958 / 114.8s |

性能闸门是**回归保护**,不是营销数字:它只要求"我们的策略在目标场景不劣于基线",
不把小样本波动当成成果。真实结论仍以 `docs/notes/experiments.md` 的多 seed 表为准。

## 效果标准(机制层)

| 能力 | 最低交付标准 | 证据 |
|---|---|---|
| 阶段驱动资源决策 | 非 GPU 阶段不得持有 ACTIVE 上下文(I1),由 property test 守住 | `tests/property/test_invariants.py` |
| 处置权租约 | 四种处置可执行;不支持时显式降级并记录 | `backends/mock.py`、`openai_compat.py` |
| KV 交接 | 交接链零重算;两种拓扑 5/5 seed 零失败 | `tests/integration/test_kv_handoff.py`、实验 W9/W10 |
| 隔离配额 | 超配额压缩而非失败;父任务不被拖死 | `tests/integration/test_nested_handoff.py` |
| 故障路径 | 工具失败不致命;取消零租约泄漏;引擎挂起有超时 | `tests/integration/test_faults.py`、`test_runtime_lifecycle.py` |
| 可观测性 | 任意运行可回答"这个任务为什么慢" | `demo trace` / `demo timeline` |

## 方式标准(工程层)

- **重构零行为变化**:拆分 `runtime.py`(644→320 行)后同一工作负载数字逐位不变。
- **数字分级**:任何表格必须标 `[L1]/[L2]`;禁止把仿真数字写成真实 GPU 测量。
- **负结果保留**:B2c 成本模型、copy 模式都被记录为负结果,没有从文档里消失。
- **状态先收敛再拆分**:K15 的教训——先有 `UnitLedger`,才允许拆文件。
- **公开面受保护**:`__all__` 由 `tests/unit/test_app.py` 守住,删符号即 breaking change。

## 已知不满足(必须如实标注)

- **L3 真实 GPU 验证缺失**(issue #3)。所有性能数字来自 mock 仿真;适配器已就绪,
  但"在真实 vLLM 上更快"这句话目前**没有证据**。
- 多机/分布式不在范围。
- `autopilot` 是安全默认,不是每个场景的最优解(见场景矩阵)。

## 复现一切

```bash
python scripts/delivery_check.py          # 交付闸门
python scripts/run_experiments.py         # 实验矩阵
python -m phaseweaver.demo scenarios      # 四类场景对比
```
