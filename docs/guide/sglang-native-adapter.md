# SGLang 原生适配器契约（分支草案，未实测）

证据等级：接口形态来自 SGLang 官方文档（native API、HiCache / RadixAttention），
**未在真实 SGLang 服务上验证**。本分支不产出 `[L3]` 结论，也不声称任何性能收益。

## 现状与缺口

`main` 没有 SGLang 后端。若直接复用 `OpenAICompatBackend`，只能走 `/v1/chat/completions`，
以下引擎特有信息全部丢失：RadixAttention 前缀树是否命中、KV 内存池实际大小、HiCache 分级状态、
原生 `/generate` 的会话语义。这正是 A 路线判停后「交给引擎原生 prefix caching 承接」所需的观察面。

## 原语映射

| SGLang 原语 | 形态 | PhaseWeaver 映射 | 边界 |
|---|---|---|---|
| `GET /server_info` | 原生 HTTP | 能力探测 | 合并了 `get_server_args` / `get_memory_pool_size` / `get_max_total_num_tokens`；可用真实 KV 池大小替代手填 `capacity_bytes` |
| `POST /generate` | 原生 HTTP（`text` / `sampling_params`） | 推理数据面 | 与 OpenAI 兼容层不同的请求形状，需单独实现 `complete()` |
| `POST /flush_cache?timeout=` | 原生 HTTP | 前缀缓存失效 | 全局副作用；HiCache 异步 in-flight 时 `timeout>0` 等待 idle，否则 400 |
| `GET /get_model_info` | 原生 HTTP | 模型能力探测 | 模型/tokenizer/架构信息 |
| `GET /health_generate` | 原生 HTTP | 就绪探测 | 会真的生成一个 token，不用于高频探测 |
| RadixAttention 前缀树 | 引擎内部（默认启用） | 引擎原生前缀共享 | 承接已判停的 A 路线；控制面不自建 refcount |
| `--enable-hierarchical-cache`（HiCache L2/L3） | 服务端启动参数 | `OFFLOAD` 分级 | 需配合存储后端；未启用时降级 `DROP` |
| 多轮 session / rid | frontend 与服务端会话语义 | `handoff` / 前缀复用 | HTTP 层参数形态待按目标版本核对 |
| `POST /update_weights_from_disk` | 原生 HTTP | 模型切换 | 触发 radix cache 自动 flush，与 park 生命周期互斥 |

## 语义边界

- RadixCache 的 block 级引用计数在引擎内部。适配层只能报告「引擎启用 / 未启用 + 内存池大小」，
  不能复制一份账本，也不能从 HTTP 层推断单次前缀命中。
- `/flush_cache`、`/update_weights_from_disk` 是进程级副作用，影响所有租户；保留显式 opt-in 并在实验报告披露。
- 没有 HiCache 或存储后端时 `supports_offload=False`，能力协商把 `OFFLOAD → DROP`，与 `main` 行为一致。
- 原生端点路径在不同版本间有合并与重命名（如 `get_server_args` 已并入 `/server_info`），
  落地时须按目标版本核对；本文件不替代官方文档。

## 待验证项

| 项 | 无 GPU 可做 | 需真机 |
|---|---|---|
| 原生端点路径 / method / 请求体与降级路径 | fake HTTP server 契约测试 | — |
| `/server_info` 内存池字段到 `BackendCaps.capacity_bytes` 的映射 | 契约测试（固定响应样本） | — |
| 取消 / 异常 / 超时三条清理路径 | 契约测试 | — |
| RadixCache 命中率、HiCache L2/L3 命中与 TTFT | — | ≥24 GiB 单卡 |
| 前缀共享对扇出负载的容量放大 | — | 起引擎后 |

## 关联

`docs/guide/extending.md`（适配器协议）、`src/phaseweaver/backends/base.py`（`InferenceBackend`）、
`CHANGELOG.md` Known issues 中 A 路线的终审、issue #3（L3 真机验证）。
