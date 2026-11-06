"""Python code executor: isolated subprocess fed through a wrapper script."""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Any

from phaseweaver.tools.base import ToolResult, ToolSpec, now
from phaseweaver.tools.proc import run_process

_WRAPPER = textwrap.dedent(
    """
    import contextlib
    import io
    import sys

    code = sys.stdin.read()
    buf = io.StringIO()
    scope = {"__name__": "__main__"}
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<tool>", "exec"), scope)
    sys.stdout.write(buf.getvalue())
    """
).lstrip()


class PythonExecutor:
    name = "python"

    async def run(self, args: dict[str, Any], spec: ToolSpec) -> ToolResult:
        code = str(args.get("code", ""))
        started = now()
        outcome = await run_process(
            [sys.executable, "-I", "-c", _WRAPPER],
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            stdin_text=code,
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
