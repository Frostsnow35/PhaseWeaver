# ADR-0001:阶段与推理上下文正交建模

- 状态:已接受
- 日期:2026-11-07

## 背景

Agent 的执行阶段(在推理/在跑工具/在等待)与其在显存中的 KV 上下文状态,如果合并成单一状态机,会立刻陷入表达困境:Agent 在 `ACTING` 时,"它不该占 GPU" 与 "它随时可能继续推理" 同时成立;而"上下文从 ACTIVE 变成 RETAINED"并不改变执行阶段。

## 决定

用两个彼此正交的维度建模:
- `Phase`:以单元为粒度,描述"它在干什么";
- `InferenceContext.state`:以上下文为粒度,描述"GPU 上还占着什么"。

两者通过不变量 I1/I2 耦合:`phase ∈ {ACTING, WAITING} ⟹ context ≠ ACTIVE`,`context ∈ {ACTIVE, RETAINED} ⟺ 持有 KV 租约`。

## 后果

- 正面:"让出 GPU" 从一句愿景变成可断言、可 property test 的约束;策略只作用于上下文维度,阶段机保持简单。
- 负面:两套状态需要各自的合法性检查,`InvariantChecker` 成为必经路径;状态快照需同时携带两者。
- 约束:任何把二者合并的简化(例如给 Phase 加 `ACTING_WITH_CACHE`)都应被拒绝,并引用本 ADR。
