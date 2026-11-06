# ADR-0002:租约抽象为"处置权 + 成本模型"

- 状态:已接受
- 日期:2026-11-07

## 背景

最初的灵感写法是"协调 KV Cache 的保留、释放、卸载及恢复"。但主流推理引擎并不暴露"把某个请求的 KV 块搬走再搬回"的通用句柄。vLLM 的真实控制面只有四类:

1. 请求级 `abort`(释放 blocks,之后只能重算);
2. 引擎级 `sleep(level=1|2)` / `wake_up`(权重卸载、KV 丢弃,整引擎粒度);
3. 前缀缓存(`reset_prefix_cache()`、automatic prefix caching)与 `kv_events` 事件流(BlockStored/BlockRemoved);
4. KV Connector(卸载到 CPU/磁盘,如 LMCache)。

## 决定

租约不表示"我借你一块可搬动的内存",而表示"我对某个推理上下文的处置权 + 该处置的代价"。
四种处置:`RETAIN`(不动,继续占租约)、`DROP`(abort,代价为按 token 重算)、`OFFLOAD`(经 connector 移出,代价为字节搬运)、`RELEASE`(终态放弃)。

适配器通过 `BackendCaps` 声明支持哪些处置及其单位成本;不支持则静默降级(`OFFLOAD→DROP`,`RETAIN→DROP`),并在事件中记录 `degraded_from`。

## 后果

- 正面:同一套语义可覆盖 vLLM / SGLang(radix cache + HiCache)/ 任意 OpenAI 兼容端点;不会写出"只在某个引擎版本成立"的耦合代码。
- 负面:调度器无法做"细粒度最优搬运",只能选策略;`OFFLOAD` 的收益依赖部署是否配置 connector。
- 约束:任何需要引擎内部句柄的新能力,必须先扩展 `BackendCaps` 与 `park/restore` 协议,不得让 `policy` 直接触碰引擎。
