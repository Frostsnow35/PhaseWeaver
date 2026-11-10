# vLLM 原生适配器契约（分支草案，未实测）

证据等级：接口形态来自 vLLM 官方文档（sleep mode、prefix caching、KV connector），
**未在真实 vLLM 服务上验证**。本分支不产出 `[L3]` 结论，也不声称任何性能收益。

## 本分支已实现（`[L3]-pending`）

| 文件 | 内容 |
|---|---|
| `src/phaseweaver/backends/vllm.py` | `VLLMBackend` 改为 POST 原生端点；`probe()` 能力探测；`VLLMCapabilities` / `capability_snapshot()` |
| `tests/integration/test_vllm_native.py` | 12 项 fake HTTP server 契约测试，零引擎依赖 |

覆盖：sleep level 1/2、wake_up tags 分步、level 2 的 `reload_weights` + `reset_prefix_cache` 顺序、
`/is_sleeping` 纯文本与 JSON 解析、`/metrics` 原文抓取、探测失败保守降级、
`OFFLOAD → DROP`、`allow_engine_sleep` / `allow_collective_rpc` 显式开关、
非法 level/tags 本地拒绝，以及取消 / 异常 / 超时后适配器状态不残留。

## 可扩展性收口（与 `main` 共用，非分支私有）

| 维度 | 收口前 | 收口后 |
|---|---|---|
| HTTP 传输 | 适配器内自写 `_request` 一份 | 复用 `backends/http_admin.py` 的 `AdminHttpClient`，与 SGLang 分支、`OpenAICompatBackend` 同一套 auth / 超时 / HTTP 状态 / JSON 错误策略 |
| 错误类型 | `VLLMAdminError(RuntimeError)` 与其他引擎互不兼容 | `VLLMAdminError(BackendAdminError)`，上层可按统一基类捕获管理面故障；对外仍是 `RuntimeError` 子类 |
| 能力快照 | 无 schema 的扁平 dict，字段来源靠人记 | `CapabilityReport`：每个字段带 `_origin`（`probed` / `declared` / `estimated` / `unknown`），附 `contract_version=vllm-admin/1` 与 `engine_version_checked` |
| 资源预算 | 声明的 `kv_pool_bytes` 直接写 `caps().capacity_bytes` | 默认只写进报告；只有 `accept_declared_capacity=True` 才进入租约预算 |
| 探测证据 | `/is_sleeping` 返回 200 就置 `admin_endpoints=True` | 必须返回合法 bool / JSON，空 body 或 HTML 一律 `probed=False`；`admin_endpoints` 的 detail 明确标注「推断，不证明 `/sleep` 可用」 |

`contract_version` 在端点形状变化时递增；`engine_version_checked` 默认 `unverified`，
真机核对后由调用方传入具体发行版。

## 现状与缺陷（`main` 基线）

`main` 的 `VLLMBackend` 继承 `OpenAICompatBackend`，只加 `/sleep?level=1` 与 `/wake_up`：

| 问题 | 事实 |
|---|---|
| 方法错误 | `_get()` 用 `urllib.request.urlopen(url)` 发 GET；官方端点是 `POST /sleep`、`POST /wake_up` |
| 缺少启用条件 | 管理端点需 `VLLM_SERVER_DEV_MODE=1` 且服务以 `--enable-sleep-mode` 启动 |
| 语义错配 | level 1 卸载权重到 CPU 并丢弃全部 KV cache，level 2 连权重也丢弃；这是引擎全局 `RELEASE`，不是按上下文 `OFFLOAD` |
| level 2 流程不全 | 唤醒后必须 `collective_rpc("reload_weights")` 再 `wake_up?tags=kv_cache`，并 `POST /reset_prefix_cache` |
| 无能力探测 | `supports_engine_sleep` 由配置写死，不探测服务端是否真的开放了端点 |

以上五项已在本分支修正或如实登记：前四项进代码与契约测试，第五项改为「探测成功且显式 opt-in 才置位」。

## 本版明确不做

- 不注入 `kv_transfer_params`，即不实现 per-request `OFFLOAD` 数据面；声明了 connector 也仍降级 `DROP`，`VLLMCapabilities.supports_offload` 恒为 `False`。
- 不统计 prefix cache 命中、不改写 `recompute_tokens`；命中率只能由引擎 `/metrics` 提供，HTTP 层拿不到 block 级信息。
- 不修改 `main` 的 `BackendCaps`；能力通过 `VLLMBackend.capability_snapshot()` 暴露，待真机验证后再决定是否提升为公共契约。
- 不把声明的 `kv_pool_bytes` 直接当资源预算：它是 `DECLARED`，默认只进报告，不参与租约决策（显式 `accept_declared_capacity=True` 除外）。

## 原语映射

| vLLM 原语 | 启用条件 | PhaseWeaver 映射 | 代价 / 边界 |
|---|---|---|---|
| `POST /sleep?level=1\|2` | dev mode + `--enable-sleep-mode` | `RELEASE`（引擎全局） | 唤醒后必然重算，记 `recompute_tokens` |
| `POST /wake_up?tags=weights\|kv_cache` | 同上 | 恢复流程 | level 2 需按 tags 分步唤醒 |
| `GET /is_sleeping` | 同上 | 状态探测 | 全部组件唤醒前返回 true |
| `POST /reset_prefix_cache` | 同上 | 前缀缓存失效 | level 2 唤醒后必做 |
| `POST /collective_rpc` | 同上 | 权重重载 | 仅限可信网络，默认不启用 |
| `--enable-prefix-caching` | 服务端启动参数 | 引擎原生前缀共享 | 承接已判停的 A 路线；控制面不自建 refcount |
| `--kv-transfer-config`（KVConnector / LMCache / Mooncake） | 服务端 + 连接器 | `OFFLOAD` | 唯一能真正搬 KV 的路径 |
| `kv_events`（ZMQ 订阅） | `--kv-events-config` | `engine_events` | 与 mock 同一套事件词汇 |
| `/metrics` 抢占计数 | Prometheus | 观测 | 用于真机校准 `thrash` 口径 |

## 语义边界

- sleep / wake_up / flush 类端点是**进程级副作用**，影响该实例上所有租户；保留显式 opt-in，任何实验报告都要披露。
- 没有 KV connector 时 `supports_offload=False`，能力协商把 `OFFLOAD → DROP`，与 `main` 行为一致，不假装搬过数据。
- RadixCache / prefix caching 的 block 级引用计数在引擎内部，HTTP 适配层拿不到；能力协商只报告「引擎启用 / 未启用」，不复制一份账本。
- 端点方法、参数与版本相关，落地时须按目标 vLLM 版本核对；本文件不替代官方文档。

## 待验证项

| 项 | 状态 |
|---|---|
| 端点 method / path / query 与降级路径 | 契约测试已覆盖（无 GPU） |
| 能力探测失败时的保守降级 | 契约测试已覆盖 |
| 取消 / 异常 / 超时三条清理路径 | 契约测试已覆盖；`asyncio.to_thread` 的取消不中断底层线程，是 stdlib 限制，已如实记录 |
| sleep↔wake 前后 TTFT、显存峰值、prefix cache 命中率 | 需 ≥24 GiB 单卡，未做 |
| `kv_events` 与本地租约账本是否一致 | 需起引擎，未做 |
| 目标 vLLM 版本的端点形状核对 | 按官方文档设计，尚未对具体发行版校验 |

## 关联

`docs/guide/extending.md`（适配器协议）、`src/phaseweaver/backends/base.py`（`InferenceBackend`）、
`CHANGELOG.md` Known issues 中 A 路线的终审、issue #3（L3 真机验证）。
