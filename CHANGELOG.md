# Changelog

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 双语 README:`README.md` 英文、`README_cn.md` 中文;顶部按 `deepseek-ai/DeepSeek-Harness` 形态——标题下紧跟 `English | [中文](...)`,当前语言纯文本、另一语言可点,无居中 div / 顶部 badges(badges 移至文末);9 个节标题仍带指向对方语言同名锚点的定点跳转链接,切换后落在同一节。`pyproject.toml` 的 `readme` 仍指向 `README.md`(PyPI long_description 为英文)。两份 Status 节更新为 `[L3] micro-benchmark` 已标定、端到端真机验证仍未完成。
- `pyproject.toml` 增加 `[project.scripts] phaseweaver = "phaseweaver.demo:main"`;`demo` 新增 `replay` 子命令与 `compare --prefix-hit-ratio`。
- `src/phaseweaver/replay.py`:定义 JSONL trace 格式(`reason` / `tool` 事件 + 可选 `prefix_id` / `prefix_tokens`),`load_trace` / `prefix_overlap` / `replay_compare`;用户可用自己的 Agent 轨迹对比 pin-always 与 B2,输出强制带 `[L1]` 标头与 seeds / C / SLO 脚注。
- `MockBackend` 增加 `prefix_hit_ratio`(`[L2]` 假设性上界,默认 0.0 时逐位保持原模型);`Workload` 与 `demo compare` 同步暴露该自变量。
- `scripts/probe_prefix.py`:`[L2]` 前缀命中敏感性闸门(h ∈ {0,.25,.5,.75,1} × 6 负载 × 3 seeds × B0/B2),原始行写 `runs/prefix-probe.json`(不入库)。
- `scripts/probe_capacity.py`:`[L2]` 容量放大上界闸门(等效容量 `C*(1+k)`,零内核改动),原始行写 `runs/capacity-probe.json`。
- `scripts/make_kaggle_notebook.py` + `scripts/kaggle_cells.py` / `kaggle_cells_late.py`:生成自包含的 Kaggle T4 x2 参数标定 notebook v2(7 个 cell,纯 PyTorch + Transformers,不起 vLLM/SGLang);产物落在 `_local/`(不入库)。cell 源码按 AGENTS.md §51 拆成两个 <400 行模块。
- `scripts/probe_calibration.py`:读 `calibration.json`,用真机常数重跑 A 的两条 `[L2]` 闸门(前缀命中增益 / 容量放大增益),对 prefill / decode / 带宽 / α 做 6 变体鲁棒性网格;`run_one(backend_caps=...)` 为回灌入口,默认 `None` 时逐位保持原行为。原始行写 `runs/calibration-probe.json`(不入库)。
- `examples/traces/sample.jsonl`:12 单元合成样例轨迹;`.gitignore` 增加 `!examples/traces/*.jsonl` 例外。
- 拆分 `demo.py`:工作负载模型与 `run_one` 留在 `demo.py`(243 行),argparse / 表格 / 子命令分派移到新增 `cli.py`(326 行),消除 >400 行违规;`main()` 保留为薄壳,`python -m phaseweaver.demo` 与 console script 行为不变(同一负载数字逐位一致)。
- `scripts/delivery_check.py` 的 lint 闸门先试 PATH 上的 `ruff`,再回退 `python -m ruff`;`docs/quality/delivery.md` 同步测试数与偶发子进程测试的复跑口径。
- **K6**:`oracle.py` 重写为与 α 吞吐模型一致的有效下界(设备级 + 单任务关键路径取大),
  `demo optimal` 输出 `bound_kind=alpha_lower_bound`。
- **K12**:KV 交接(handoff)——fork 时父单元把 KV 所有权移交子单元,join 时交回,零重算;
  `inherit_mode = none | handoff`。
- **M5**:`OpenAICompatBackend`(stdlib HTTP)与 `VLLMBackend`(可选引擎级 sleep/wake),
  实现与 mock 相同的 `InferenceBackend` 协议;能力协商会把 `OFFLOAD` 自动降级为 `DROP`。
- 仓库:实验记录、开发日志、取舍记录(ADR)移出仓库,只保留在本地(`_local/`);仓库只留可复现脚本、契约文档与 `AGENTS.md` 的 PR 规范。
- 子 Agent 扇出:`Runtime.fork()` 现在会让父单元先 park 上下文并进入 `WAITING`(满足 I1),
  join 后回到 `REASONING`;并发 fork 用深度计数保证阶段正确。
- 工作负载支持 `fanout` / `sub_rounds`,新增 `examples/workloads/w5-fanout.toml`。
- `scripts/push.ps1`:本机沙箱下的推送脚本(openssl 后端 + 令牌头,不落盘)。
- **K8**:`LeaseBroker` 支持 scope 配额,`fork(mode=ISOLATE)` 为子树开硬上限;
  新增 `examples/workloads/w5-isolate.toml`。
- **K3(实验)**:`SchedulerConfig.use_cost_model` 提供"容量代价 vs 重算代价"的决策器,默认关闭。
- 指标新增 `jain_fairness`,用于观察不均衡负载下的公平性。
- `examples/workloads/w6-greedy*.toml`:不均衡/贪婪子 Agent 的对抗负载。

### Changed

- 实验结论:成本模型(B2c)显著劣于 slack 启发式,默认不启用,记录为负结果。
- **ISOLATE 语义**:子单元因配额失败时不再杀死父任务,异常在 `fork()` 边界内被收敛。

### Removed

- `site/playground.html` 与 `scripts/build_playground_data.py`:交互实验场移出范围,两份 README 与站点首页/页脚的入口链接一并移除。零安装试用改由 `phaseweaver replay`(`docs/guide/quickstart.md`)承载;Pages 上的 `/playground.html` 会 404。负结果「27/36 网格点 B2 与 B0 一致」保留在 Known issues。

### Known issues

- K6:B4 下界与 α 吞吐模型不自洽,optimality gap 暂不可引用。
- K11:被配额拒绝的子单元没有降级重试。
- **[L2] 前缀缓存路线(A)判停**:`h=0 → 0.5` 时 B2 的 goodput 增益为 −1.7%–+2.3%(6 负载 × 3 seeds),0/6 通过 5% 闸门;即使 `h=1.0` 完全消除重算,6 个负载增益仍 < 3%。判据与原始行见 `scripts/probe_prefix.py` / `runs/prefix-probe.json`。该探针只建模「重算 prefill 时间变便宜」。
- **[L2] 容量放大补齐,A 的终审**:`scripts/probe_capacity.py` 用等效容量 `C*(1+k)` 扫上界。`k=0.5`(共享前缀约 33%,真实 Agent 的量级)时扇出/排队负载显著受益:w3 +25.1%、w5 +42.3%、w9 +48.4%;但 w2 +0.5%、w8 +6.3%、w10 +6.9%。**结论:A 想解决的问题在特定负载上确实值钱,但解决方式不该是自己写 refcount 账本**——vLLM prefix caching / SGLang radix cache 已实现 paged-block 去重,自己实现既违背规格 §2.2「不改引擎内核」,也必然拿不到 block 级真实状态。改由「能力协商 + 引擎原生 prefix caching」承接,作为真机实验的一个臂。
- Kaggle T4 x2(sm_75)无 bf16 / 无 FA2,vLLM/SGLang 多卡在该环境不稳定(NCCL / FlashInfer / libcuda),**跑不了战略 v2 的 SGLang+HiCache 对标**。真机首轮目标收窄为 `[L3] micro-benchmark` 参数标定,不得表述为端到端性能结果。
- **[L3] 首轮真机标定(Kaggle T4 x2,2026-11-10)**:`bytes_per_token=57344` 与 Qwen2.5-7B 的 config 精确对齐;PCIe 实测 d2h 13.02 / h2d 12.22 GB/s(mock 12.88 GB/s,−5%);decode 33–34 tok/s(mock 45,高估 36%);batched decode 拟合 `contention_alpha≈0.0175`(mock 0.15,高估约 8 倍)。口径为 `[L3] micro-benchmark / parameter calibration`,非端到端结果;提取的数据在 `runs/kaggle-l3/`(不入库)。
- **v1 标定数据三处无效项,不得引用**:① Cell 2 把首次 CUDA kernel 初始化的 256-token 前向计入(164 tok/s),5 长度取均值造出 4152 tok/s;去冷启动后按长度为 6372 / 7109 / 4552 / 2562 tok/s,`t(4096)/t(1024)=11.1`(4× token 线性应为 4.0),常数 prefill 模型不成立。② Cell 4 的 `for t in held: del t` 只解绑循环变量、张量从未释放,得到 105 / 0 / 26 / 0 的非单调曲线,且未预留权重显存。③ Cell 6 把空闲显存 13.0 GiB 写成 `capacity_bytes`,但 mock 的 capacity 是 KV 预算;7B fp16 权重约 14.2 GiB,在单卡 14.56 GiB 上几乎放不下。notebook v2 已修三处:warmup + 3 次取中位 + 按长度曲线、`held.clear()` + 权重预留、不再产出单一 prefill 常数。
- **A 终审补强**:`scripts/probe_calibration.py` 用上述真值及其上下界跑 6 个变体(3 seeds × 6 负载),前缀命中闸门 0–1/6 通过判停,容量放大闸门 4–5/6 通过、平均增益 21.6%–32.4%;**0/6 变体满足「两闸门同时翻转」的重开条件**,自建 refcount 账本路线维持关闭,容量收益仍由引擎原生 prefix caching 承接。口径限制:速率来自 0.5B、`bytes_per_token` 来自 7B,不是同一份 `BackendCaps`,只用作鲁棒性边界,不回填。
- B2 在容量不紧张时全程 `RETAIN`、与 B0 完全一致(原 36 点网格中 27 个如此);这是设计行为,不作为收益呈现。该网格随 playground 移除,两条闸门仍可由 `scripts/probe_prefix.py` / `probe_capacity.py` 复现。

### 计划中

- K11:被配额拒绝的子单元支持降级重试(缩小上下文)
- K6:按 α 共享吞吐模型重做 B4 下界
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
