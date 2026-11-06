# ADR-0005:优化目标为 SLO 达成率 + 尾部延迟不劣化

- 状态:已接受
- 日期:2026-11-07

## 背景

初版设计把"让出 GPU"做成逐单元的局部规则(工具时长 < 2 s 就保留),并以 `goodput` 与平均延迟作为判据。这有两个问题:

1. **局部规则在过载下会同步振荡**:所有 Agent 同时进入工具期 → 集体 park;同时返回 → 集体 restore,造成抖动风暴,尾延迟比"什么都不做"更差。
2. **只报均值会掩盖伤害**:纯吞吐目标可以靠牺牲长尾换数字,与"性能稳定"直接冲突。

## 决定

1. 目标函数显式写成 SLO-aware 形式:
   `maximize Σ_i w_i · 1[task_i 在 SLO_i 内完成]`,`s.t. Σ resident_kv ≤ C - headroom, concurrent_reasoning ≤ R_max`。
2. 判据要求 **`p99` 与 `thrash_rate` 同时不劣化**,否则即使 `goodput` 上升也不认定成功。
3. 决策从逐单元局部规则升级为消费只读快照的 `SystemScheduler`;局部规则保留为 B2a 消融项。
4. 稳定性做成四个可执行旋钮(`min_residency_s`、`park_cooldown_s`、`hysteresis_band`、`restore_backpressure`),各自有独立消融实验。
5. 公平性用 max-min fair + DRR,并以不变量 I8(防饥饿)固化。
6. 增加离线最优上界 B4(`oracle.py`)计算 optimality gap,回答"系统层面到底多接近最优"。

## 后果

- 正面:结论不会被均值掩盖;抖动与公平性成为一等公民;有 gap 数字支撑"系统层面较优"的说法。
- 负面:实验矩阵显著变大(E1–E5 + 6 组消融 + 20 seed),M3/M4 工作量上升;需要额外的 `oracle.py` 与滑动窗口指标实现。
- 负面:过载下若 `p99` 确实劣化,就必须如实报告——这是被明确接受的代价,不允许事后改判据。
- 约束:`est_remaining_work` 用简单估计器(历史 P50 / 轮数默认),不引入学习模型;估计器可替换,并作为消融维度之一。
