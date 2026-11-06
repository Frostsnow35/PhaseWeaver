"""Tool supervision: track live tool tasks per unit and cancel them on demand.

Two modes:
- simulated (default for experiments): the tool consumes *virtual* time only, so a
  run is fully deterministic and the tool phase actually costs what the workload
  says. No subprocess is spawned.
- real (for a live engine): runs the executor and awaits it for real.
"""

from __future__ import annotations

import asyncio
from typing import Any

from phaseweaver.clock import Clock
from phaseweaver.journal import EventJournal
from phaseweaver.tools.base import ToolExecutor, ToolResult, ToolSpec, ToolStatus, now
from phaseweaver.tools.browser_exec import BrowserExecutor
from phaseweaver.tools.python_exec import PythonExecutor
from phaseweaver.tools.shell_exec import ShellExecutor


class ToolSupervisor:
    def __init__(
        self,
        journal: EventJournal,
        clock: Clock | None = None,
        *,
        simulated: bool = False,
    ) -> None:
        self._journal = journal
        self._clock = clock
        self._simulated = simulated
        self._executors: dict[str, ToolExecutor] = {
            e.name: e for e in (PythonExecutor(), ShellExecutor(), BrowserExecutor())
        }
        self._live: dict[str, set[asyncio.Task[ToolResult]]] = {}

    async def run(self, unit_id: str, spec: ToolSpec, args: dict[str, Any]) -> ToolResult:
        self._journal.append(
            "tool_started",
            unit_id=unit_id,
            payload={"tool": spec.name, "executor": spec.executor},
        )
        if self._simulated:
            result = await self._run_simulated(spec)
        else:
            result = await self._run_real(unit_id, spec, args)
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

    async def _run_simulated(self, spec: ToolSpec) -> ToolResult:
        duration = spec.expected_duration_s if spec.expected_duration_s is not None else 1.0
        started = now()
        if self._clock is not None and duration > 0:
            await self._clock.wait(duration)
        return ToolResult(
            status=ToolStatus.OK,
            stdout="",
            duration_ms=int(duration * 1000),
            started_at=started,
            finished_at=now(),
            meta={"simulated": True},
        )

    async def _run_real(
        self, unit_id: str, spec: ToolSpec, args: dict[str, Any]
    ) -> ToolResult:
        executor = self._executors[spec.executor]
        task: asyncio.Task[ToolResult] = asyncio.create_task(executor.run(args, spec))
        self._live.setdefault(unit_id, set()).add(task)
        try:
            return await task
        finally:
            self._live[unit_id].discard(task)

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
