# AGENTS.md — Agent 协作与 PR 规范

本文件约束由 Agent(或人类协作者)提交到本仓库的改动。**PR 是唯一的记录载体**:
仓库只保存可执行的代码、可复现的脚本与稳定的契约文档;过程性记录不进入版本库。

## 1. 什么不进仓库

| 类别 | 举例 | 存放位置 |
|---|---|---|
| 实验记录 | 多 seed 结果表、扫描数据、负结果分析 | `_local/notes/`(已 gitignore) |
| 开发日志 | 会话式开发记录、教训、作废范围 | `_local/notes/` |
| 取舍记录(ADR) | 关键设计决策的来龙去脉 | `_local/adr/` |
| 运行产物 | `runs/*.jsonl`、时间线导出、截图 | `runs/`(已 gitignore) |
| 环境与缓存 | `.venv/`、`.uv-cache/`、各测试缓存 | 本地,不提交 |

规则:**结果写进 PR 描述,记录留在本地**。提交前用 `git status --short` 自查,
不应出现上述任何文件。

## 2. PR 必须包含的内容(在描述里,不在文件里)

1. **变更摘要**:改了什么、为什么、影响哪些不变量或契约。
2. **复现命令**:从干净环境可复制的完整命令(含容量、seed、策略)。
3. **结果表**:标注证据等级 `[L1]` 仿真 / `[L2]` 半真实 / `[L3]` 真实 GPU;
   必须写明 seeds、容量、SLO 口径。没有数字的“性能提升”不予合并。
4. **负结果**:失败或未采用的实验也要写进 PR,并在摘要里说明不采用的原因。
5. **风险与回滚**:最坏情况、如何回退(通常 revert 本 PR)。
6. **自查清单**:使用 `.github/pull_request_template.md`。

## 3. 分支与提交

- 分支名:`agent/<主题>`,如 `agent/fix-reaper-false-gone`。
- 提交信息:Conventional Commits(`fix:` / `feat:` / `refactor:` / `docs:` / `chore:`)。
- 一个 PR 只做一件逻辑变更;重构与行为变更分开。
- 不直接向 `main` 提交(仅限仓库维护者的紧急修复,且事后补 PR 说明)。

## 4. 合并前必须通过的闸门

```bash
uv run pytest -q
uv run ruff check .
uv run mypy --platform linux src/phaseweaver
uv run mypy --platform win32 src/phaseweaver
python scripts/delivery_check.py --quick   # 六项交付闸门的本地版本
```

- CI 四格(ubuntu/windows × py3.11/3.12)必须全绿。
- 改动站点时,确认 Pages 部署成功(`Deploy demo site` 工作流)。

## 5. 代码规范

- 单文件 ≤ 400 行;超出先拆模块,再提交。
- 新增状态转移必须同时补单元测试与不变量断言。
- 新增清理路径必须覆盖“取消/异常/超时”三种情况。
- 不新增 `TODO` / `FIXME` / `type: ignore`(确有必要需在 PR 里说明理由)。
- `__all__` 的删除或重命名视为 breaking change,需在 `CHANGELOG.md` 记录。
- 不隐藏负结果;不为了让数字好看而调参,也不删已有的负结果复现路径。

## 6. 文档更新规则

| 改动类型 | 需要更新 |
|---|---|
| 系统契约、状态机、事件类型 | `docs/specs/` |
| 扩展点、协议、插件用法 | `docs/guide/` + `examples/` |
| 质量门、交付标准 | `docs/quality/delivery.md` |
| 站点页面 | `site/index.html`(触发 Pages 部署) |
| 用户可见行为 | `README.md` + `CHANGELOG.md` |

## 7. 禁止事项

- 不提交任何凭据、token、本地路径或截图产物。
- 不 force-push 受保护的 `main`。
- 不在仓库内新增实验/日志/ADR 文件(见 §1)。
- 不把仿真数据表述为真实 GPU 结果。
