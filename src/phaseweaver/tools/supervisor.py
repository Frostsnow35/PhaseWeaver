"""Tool supervision: track live tool tasks per unit and cancel them on demand."""

from __future__ import annotations

import asyncio
from typing import Any

from phaseweaver.journal import EventJournal
from phaseweaver.tools.base import ToolExecutor, ToolResult, ToolSpec
from phaseweaver.tools.browser_exec import BrowserExecutor
from phaseweaver.tools.python_exec import PythonExecutor
from phaseweaver.tools.shell_exec import ShellExecutor


class ToolSupervisor:
    def __init__(self, journal: EventJournal) -> None:
        self._journal = journal
        self._executors: dict[str, ToolExecutor] = {
            e.name: e for e in (PythonExecutor(), ShellExecutor(), BrowserExecutor())
        }
        self._live: dict[str, set[asyncio.Task[ToolResult]]] = {}

    async def run(self, unit_id: str, spec: ToolSpec, args: dict[str, Any]) -> ToolResult:
        executor = self._executors[spec.executor]
        self._journal.append(
            "tool_started",
            unit_id=unit_id,
            payload={"tool": spec.name, "executor": spec.executor},
        )
        task: asyncio.Task[ToolResult] = asyncio.create_task(executor.run(args, spec))
        self._live.setdefault(unit_id, set()).add(task)
        try:
            result = await task
        finally:
            self._live[unit_id].discard(task)
        self._journal.append(
            "tool_finished",
            unit_id=unit_id,
            payload={
                "tool": spec.name,
                "status": result.status.value,
                "duration_ms": result.duration_ms,
            },
        )
        return result

    async def cancel_unit(self, unit_id: str) -> None:
        tasks = self._live.get(unit_id, set())
        if not tasks:
            return
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._journal.append("tool_killed", unit_id=unit_id, payload={"reason": "cancel"})

    async def shutdown(self) -> None:
        for unit_id in list(self._live):
            await self.cancel_unit(unit_id)
