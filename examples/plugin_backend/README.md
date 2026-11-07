# 插件示例:注册一个第三方后端

这个目录演示如何把 PhaseWeaver 当作生态来扩展——不改运行时源码,只发一个包。

## 安装

```bash
cd examples/plugin_backend
pip install -e .
```

## 使用

```python
from phaseweaver import PhaseWeaver

async with PhaseWeaver(backend="example-device", profile="research") as pw:
    await pw.run("task", agent)
```

`PhaseWeaver` 通过入口点组 `phaseweaver.backends` 找到 `build_backend`,
调用它得到后端实例;此时自动切到真实时钟、关闭工具仿真。

## 契约

- 入口点值必须是**零参可调用**,返回实现 `InferenceBackend` 的对象;
- `caps()` 必须**诚实**:不支持的能力写 `False`,运行时会显式降级并记录 `degraded_from`;
- 不要 import `phaseweaver` 的内部模块(除 `backends.base` 与 `backends.mock` 这类半公开面)。

## 三组入口点

| 组 | 值 | 用途 |
|---|---|---|
| `phaseweaver.backends` | 零参工厂 → `InferenceBackend` | 接新引擎 |
| `phaseweaver.policies` | 可调用 → `Scheduler` 实例 | 换资源策略 |
| `phaseweaver.tools` | `ToolExecutor` 实例 | 加工具类型 |
