# Changelog

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划中

- K3:把 `kv_waste` 接入调度目标并做消融
- K5:定位 2 GiB 下偶发 p99 异常
- K6:B4 最优性间隙全量(E5)
- K7:子 Agent 扇出实验(`SHARE` / `ISOLATE`)
- M5:`VLLMBackend` / `OpenAICompatBackend`(需真实 GPU 产出 L3)

## [0.1.0] — 2026-11-07

### Added

- 执行单元树:Agent/Subagent/ToolCall,树形 fork/join/取消,阶段转移表与 kind 守卫。
- 双状态机:`Phase` 与 `ContextState` 正交建模,7 条不变量(I1–I4、I6、I8)可断言。
- 租约经济:`Budget` / `LeaseBroker` / 原子 `resize()`,四种处置 `RETAIN|OFFLOAD|DROP|RELEASE`。
- 调度:基线 B0/B1/B2a 与全局 SLO 调度器 B2(slack + 四旋钮防抖 + DRR)。
- mock 推理后端:虚拟时钟、block 级 KV 记账、共享吞吐竞争(α 模型)、kv_events 同词汇。
- 工具执行:Python/Shell 真执行,Windows Job Object + POSIX 进程组回收。
- 可观测:append-only JSONL 事件日志、指标重放、`kv_waste` 口径。
- 实验链路:`compare` / `burst` / `sweep` / `optimal` 四个子命令与 TOML 工作负载。
- 离线最优性下界 `oracle.py`(heuristic lower bound)。

### Fixed

- 并发任务共用同一 `unit_id` 导致状态互相覆盖。
- 工具单元缺少 `ACTING→COMPLETED` 合法转移。
- 任务结束时引擎 block 未释放,导致预算看似空闲而引擎耗尽。
- SLO 判定把时长与绝对时间戳比较,导致永远达标。
- 扩扩容死锁(所有常驻单元同时扩容无人退让)与时钟推进异常。
- 租约 resize 的空窗导致引擎块数超预算。

### Known issues

- K5:2 GiB 五 seed 中 B2 有一次 p99 异常,未归因。
- 真实 GPU 适配(M5)尚未实现,所有性能数字均为 mock([L1]/[L2])。
