"""Shell command executor with process-group cancellation."""

from __future__ import annotations

from typing import Any

from phaseweaver.tools.base import ToolResult, ToolSpec, now
from phaseweaver.tools.proc import run_process


class ShellExecutor:
    name = "shell"

    async def run(self, args: dict[str, Any], spec: ToolSpec) -> ToolResult:
        command = str(args.get("command", ""))
        started = now()
        outcome = await run_process(
            command,
            shell=True,
            timeout_s=spec.timeout_s,
            grace_ms=spec.grace_ms,
        )
        return ToolResult(
            status=outcome.status,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            exit_code=outcome.exit_code,
            duration_ms=int((now() - started) * 1000),
            truncated=outcome.truncated,
            started_at=started,
            finished_at=now(),
        )
