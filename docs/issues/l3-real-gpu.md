# L3: 在真实 GPU 上验证 vLLM 适配器与"让出"收益

## 背景

目前所有数字都是 mock 仿真([L1]/[L2]),**没有任何真实引擎测量**。
`VLLMBackend` / `OpenAICompatBackend` 已实现,并用本地假 OpenAI 服务验证了协议层
(`tests/integration/test_openai_backend.py`),但从未连接真实 vLLM。

## 为什么需要 GPU

- 需要 CUDA 设备与足够显存(建议 ≥ 24 GiB,跑 7B 级模型);
- 需要 vLLM 服务端(可选 LMCache/connector 以验证 `OFFLOAD` 路径);
- 需要观察真实 TTFT / 吞吐 / 显存占用,而不是仿真的记账值。

## 实验协议(建议)

1. 起服务:`vllm serve <7B> --max-model-len 8192 --enable-prefix-caching`,记录配置。
2. 用 `Runtime(backend=VLLMBackend(...), clock=RealClock(), config=RuntimeConfig(simulate_tools=False))`
   跑与 W3 同构的工作负载。
3. 对照组:B0(pin always)vs B2(全局 SLO 调度)。
4. 记录:TTFT、端到端 p50/p95/p99、GPU 显存峰值、`nvidia-smi` 采样、重算 token 估算。
5. 若验证 `OFFLOAD`:改用带 KV connector 的部署,并对比 `DROP` 的差异。

## 完成判据

- [ ] 真实引擎上 B2 的 p99 不劣于 B0,且 `kv_waste` 口径可计算
- [ ] 至少一张 L3 表格进入 `docs/notes/experiments.md`,表头标注 `[L3]`
- [ ] 记录一次失败/退化(例如 `OFFLOAD` 不可用时的降级)以便验证能力协商

## 备注

本 issue 不需要改运行时;如果必须改,说明 M5 的适配器抽象有缺陷,应在 PR 里一并修正。
