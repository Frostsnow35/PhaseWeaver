"""Browser executor: interface only. Phase 1 does not drive a real browser."""

from __future__ import annotations

from typing import Any

from phaseweaver.tools.base import ToolResult, ToolSpec, ToolStatus, now


class BrowserExecutor:
    name = "browser"

    async def run(self, args: dict[str, Any], spec: ToolSpec) -> ToolResult:
        ts = now()
        return ToolResult(
            status=ToolStatus.ERROR,
            stderr="browser executor not implemented (phase 1 defines the interface only)",
            started_at=ts,
            finished_at=ts,
        )
