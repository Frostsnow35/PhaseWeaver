# 评测与审查报告

- 日期:2026-11-07
- 对象:`main` @ `ac7ae69` 之后的审计分支,103 个测试
- 方法:测试套件 + 覆盖率测量 + 性能基准 + 静态审查(复杂度/死代码/异常路径)
- 原则:**先测量,再结论;发现的缺陷修掉并留下回归测试**

## 0. 结论

| 维度 | 评级 | 依据 |
|---|---|---|
| 机制正确性 | 良好 | 103 测试全绿;不变量 property test;故障注入全覆盖 |
| 工程质量 | 良好 | ruff/mypy 双平台 0 告警;无 TODO/type-ignore;0 个函数复杂度 >12 |
| 性能(仿真) | 良好 | 8 agents 0.25 s;回放 500k events/s;16 agents 墙钟 0.78 s |
| 真实 GPU 证据 | **缺失** | L3 未验证(issue #3),所有性能数字标注 `[L1]` |
| 交付闸门 | **6/6 PASS** | `python scripts/delivery_check.py` |

一句话:**能不能交付取决于是否接受"只有仿真证据";代码本身达到了交付标准。**

## 1. 测试与正确性

| 类别 | 数量 | 覆盖内容 |
|---|---|---|
| unit | 70 | 状态机、租约、调度、记账、autopilot、profile、oracle |
| property | 2 | 不变量 I1–I8 的随机路径 |
| integration | 31 | 生命周期、KV 交接(嵌套/扇出)、故障注入、reaper、观测 |

- 确定性:同 seed 两次运行 `summary_row()` 逐位一致(闸门 4)。
- 故障注入:工具失败不致命、取消零租约泄漏、引擎挂起有超时、交接失败不泄漏 depth/scope。
- 审计中**新发现并修复 2 个真实缺陷**(F1、F2),均附回归测试。

## 2. 覆盖率(full suite,coverage.py)

总体 **79%**(2612 语句,540 未覆盖)。低于 80% 的模块:

| 模块 | 覆盖率 | 原因 | 处置 |
|---|---|---|---|
| `tools/reaper.py` | 56% | 平台分支 + 权限拒绝路径 | 本轮补 4 个测试(F2) |
| `ledger.py` | 63% | 死代码 `owner_of_handle` | 已删除 |
| `policy.py` | 80% | 部分策略组合未跑 | 可接受 |
| `scenarios.py` | 0% | 只被 CLI 使用 | 由 `demo scenarios` 覆盖(冒烟) |
| `cleanup.py` 等 | 91%+ | — | — |

> 说明:coverage.py 不是项目依赖,用 `uv run --with coverage` 临时测量;不作为 CI 闸门,
> 避免为了数字写无意义测试。

## 3. 性能(仿真后端,`scripts/benchmark.py`)

**单次运行**(8 agents / 2 GiB / B2):wall **0.93 s**,进程峰值 0.3 MiB,
peak KV 1883 MiB,goodput 0.0899,p99 70.6 s。

**规模扩展**:

| agents | wall(s) | goodput | p99(s) |
|---|---|---|---|
| 4 | 0.31 | 0.0797 | 42.4 |
| 8 | 0.87 | 0.0899 | 70.6 |
| 12 | 2.57 | 0.0909 | 102.3 |
| 16 | 5.17 | 0.0823 | 139.8 |

墙钟随 Agent 数**超线性**(16/4 = 16.7×)。归因:虚拟时钟驱动在每次 tick 上唤醒全部
等待者,任务数上升后 event-loop 调度成本占比升高;**与真实引擎无关**(真实后端由解码时间主导)。
结论:仿真实验建议 ≤12 agents;需要更大规模时应改事件驱动时钟(F9,未修)。

**离线回放**:1285 events / 252 spans / 12 个 agent trace = **15.4 ms**(约 83k events/s),
线性增长;观测能力对运行时是**零开销**(纯重放)。

## 4. 代码健康

- 40 个模块,最大文件 `demo.py` 434 行(CLI 聚合器),核心 `runtime.py` 329 行。
- **0** 个函数圈复杂度 > 12(审计前有 2 个:`metrics.replay` 22、`fork` 14)。
- **0** 个 `TODO/FIXME/type: ignore` 在 `src/`。
- 死代码清理 3 处(`owner_of_handle`、`parked_lease_kind`、`_owns_clock`)。

## 5. 审查发现

| # | 严重度 | 问题 | 状态 |
|---|---|---|---|
| F1 | **高** | `fork` 在继承(handoff)抛异常时不恢复 `_fork_depth`、不关闭 ISOLATE scope,后续 fork 永久错乱 | **已修** + 回归测试 |
| F2 | **高** | `tree_is_gone` 在 `tasklist` 被拒绝时返回空输出 → 误判进程"已消失",导致孤儿进程泄漏 | **已修**(OpenProcess + CSV 精确匹配 + 保守回退)+ 4 测试 |
| F3 | 中 | `metrics.replay` 圈复杂度 22,指标 bug 易藏 | **已修**(拆 `_ReplayState`,每事件一个 handler) |
| F4 | 中 | `fork` 圈复杂度 14 | **已修**(抽 `_inherits_now` / `_park_parent_for_fork`) |
| F5 | 低 | 死代码 3 处 | **已修** |
| F6 | 低 | `scenarios` 用相对路径,依赖 CWD | **已修**(`workloads_root()` + 环境变量覆盖) |
| F7 | 低 | `run_task` 宽泛 `except` 把编程错误也记成 FAILED,不可诊断 | **已修**(`run_error` 事件含 traceback;`strict_errors` 开关) |
| F10 | 低 | `Scheduler.admit()` 从未被运行时调用(死协议,扩展点形同虚设) | **已修**:接入准入闸门(背压等待 + 超时),R6 的负结果也由此发现 |
| F8 | 低 | `_sync_lease` 用 `clock.wait(0.05)` 轮询等待容量 | **开放**(仿真下可接受;真实时钟下应改事件通知) |
| F9 | 低 | 仿真墙钟在 >12 agents 超线性 | **已修**:虚拟时钟改事件驱动,16 agents 5.17s→0.78s(6.6×),回放 15.4ms→2.7ms |

## 6. 与交付标准的差距

1. **L3 真实 GPU 验证缺失**——唯一无法在本地补齐的项,已开 issue #3。
2. `autopilot` 不是所有场景最优(见场景矩阵),文档已如实标注。
3. F8/F9 为已知技术债,不影响当前结论,已登记。

## 7. 复现

```bash
python scripts/delivery_check.py          # 6 项交付闸门
python scripts/benchmark.py               # 性能基准
uv run --with coverage coverage run --source=src/phaseweaver -m pytest
uv run --with coverage coverage report --skip-covered
python -m phaseweaver.demo scenarios --capacity-gib 2 --seeds 3
```
