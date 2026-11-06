# 实验记录

> 本文所有数字均为 **L1(机制证明,mock 仿真)** 或 **L2(参数敏感性)**,不是真实 GPU 测量。
> 环境:Windows 11 / i5-1340P / 16 GB / **无 NVIDIA GPU**;虚拟时钟,单 seed 或 20 seed 标注见各表。

## 0. 结论摘要(先说不好听的)

1. 垂直切片跑通:12 Agent × 4 轮、4 GiB 预算、4 种策略在一次运行内完成并产出指标。
2. **核心主张尚未成立**:在当前的 mock 模型下,B2/B2a 的 `goodput` 与 `p99` 都不优于 B0。
3. 已定位两个真实原因(不是调参问题):
   - **R1 计算无竞争**:`MockBackend` 只对 **KV 容量**建模,未对 **算力**建模(每条请求独占 3200 tok/s prefill、45 tok/s decode)。既然释放显存不会让任何请求变快,让出 GPU 的唯一后果就是**重算成本**,于是 B2 必然更差。
   - **R2 租约与引擎分配在 resize 窗口内会漂移**:重算/resize 时先释放旧租约再申请新租约,窗口期内引擎仍按旧块数占用 block,导致 10 个上下文同时"持块但无租约",第 10 个任务 `EngineOutOfMemory`(见 §3 的 3 个 FAILED)。

## 1. E1 稳态对比 [L1]

命令:

```
python -m phaseweaver.demo compare --workload examples/workloads/w1.toml --capacity-gib 4 --policies B0,B1,B2a,B2
```

负载:W1,12 Agent × 4 轮,context 8192 tok,out 128 tok,tool lognormal(P50=2 s),λ=0.5/s,SLO=120 s。
容量 4 GiB,seed=7(单 seed)。

| 策略 | 完成 | SLO | goodput | p50 | p95 | p99 | thrash | 重算 token | 峰值 KV |
|---|---|---|---|---|---|---|---|---|---|
| B0 全保留 | 9 | 0.750 | 0.2649 | 27178 | 28867 | 28953 | 0.00 | 0 | 4060 MiB |
| B1 TTL 驱逐 | 9 | 0.750 | 0.2566 | 27285 | 27953 | 28111 | 0.00 | 0 | 4060 MiB |
| B2a 局部让出 | **12** | **1.000** | 0.2010 | 35550 | 49405 | 55509 | 23.12 | 253952 | 4046 MiB |
| B2 全局 SLO | 9 | 0.750 | 0.1657 | 34644 | 40980 | 42406 | 12.15 | 163840 | 4053 MiB |

解读(如实):

- B2a 把完成数从 9 提到 12、SLO 达成率从 0.75 提到 1.0 —— 这是**准入侧**的收益,证明"让出显存 → 更多任务能进入"的机制方向正确。
- 但代价是 p99 从 28.9 s 涨到 55.5 s、goodput 下降 24%:重算把每个任务拖慢了,而 mock 里**没有任何请求因为别人让出而变快**。
- B2(全局)反而不如 B2a:其 `park_cooldown_s`/`allow_restore` 让等待者长时间拿不到租约,3 个任务直接失败。B2 的冷却参数在"容量极紧"场景下需要重新设计(见 §4)。

## 2. 指标可信度说明

- `idle_waste_mib_s` 在所有策略下都是 **0**:只要有一个单元在 REASONING,GPU 就不算空闲。12 Agent 并发时这一项恒为 0,说明它**不适合作为当前负载下的主指标**——应改为"单位任务的有效 KV 占用"或"KV 占用 × 该任务未在推理的时长"。
- `thrash` 在 B0/B1 上为 0 是正确的(它们从不改变上下文状态),因此在 B0 与 B1 之间比较 thrash 无意义。

## 3. 失败样本(可复现)

B2 下 12 个任务里 3 个失败,错误一致:

```
EngineOutOfMemory: blocks exhausted: used=4608 want=512 cap=4681
```

复现命令(把策略换成 B2、agents=12、capacity=4):
`python -m phaseweaver.demo compare --workload examples/workloads/w1.toml --capacity-gib 4 --policies B2`

根因 R2 见 §0。

## 4. 下一步(按优先级)

| 编号 | 事项 | 判据 |
|---|---|---|
| F1 | **租约改成 block 粒度**,与引擎共用同一整数计数;resize 只申请差额 | B2 下 0 个 `EngineOutOfMemory`;I3 仍成立 |
| F2 | **给 mock 加算力竞争**:所有 REASONING 共享总吞吐,单流速率随并发下降 | B0 在过载下 p99 显著劣化,B2 出现胜出区间 |
| F3 | **B2 准入/恢复背压重设计**:等待队列按 slack 排序,`allow_restore` 与 `park_cooldown` 解耦 | W1 下 B2 完成数与 B2a 持平且 p99 更低 |
| F4 | 主指标从 `idle_waste_bytes_s` 换成 `kv_occupancy_waste`(仅统计"占用但该任务未在推理"的时长) | 首次出现非零且能区分 B0/B2 |
| F5 | 全部重跑 E1–E5,产出 20-seed mean±std 与 Pareto 前沿 | 报告里同时给正负结果 |

在 F1–F4 完成前,**不得**把本文件的任何数字写进作品材料。
