# vLLM 原生适配器契约（分支草案，未实测）

证据等级：接口形态来自 vLLM 官方文档（sleep mode、prefix caching、KV connector），
**未在真实 vLLM 服务上验证**。本分支不产出 `[L3]` 结论，也不声称任何性能收益。

## 现状与缺陷

`main` 的 `VLLMBackend` 继承 `OpenAICompatBackend`，只加 `/sleep?level=1` 与 `/wake_up`：

| 问题 | 事实 |
|---|---|
| 方法错误 | `_get()` 用 `urllib.request.urlopen(url)` 发 GET；官方端点是 `POST /sleep`、`POST /wake_up` |
| 缺少启用条件 | 管理端点需 `VLLM_SERVER_DEV_MODE=1` 且服务以 `--enable-sleep-mode` 启动 |
| 语义错配 | level 1 卸载权重到 CPU 并丢弃全部 KV cache，level 2 连权重也丢弃；这是引擎全局 `RELEASE`，不是按上下文 `OFFLOAD` |
| level 2 流程不全 | 唤醒后必须 `collective_rpc("reload_weights")` 再 `wake_up?tags=kv_cache`，并 `POST /reset_prefix_cache` |
| 无能力探测 | `supports_engine_sleep` 由配置写死，不探测服务端是否真的开放了端点 |

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

| 项 | 无 GPU 可做 | 需真机 |
|---|---|---|
| 端点 method / path / query 与降级路径 | fake HTTP server 契约测试 | — |
| 能力探测失败时的保守降级 | 契约测试 | — |
| 取消 / 异常 / 超时三条清理路径 | 契约测试 | — |
| sleep↔wake 前后 TTFT、显存峰值、prefix cache 命中率 | — | ≥24 GiB 单卡 |
| `kv_events` 与本地租约账本是否一致 | — | 起引擎后 |

## 关联

`docs/guide/extending.md`（适配器协议）、`src/phaseweaver/backends/base.py`（`InferenceBackend`）、
`CHANGELOG.md` Known issues 中 A 路线的终审、issue #3（L3 真机验证）。
