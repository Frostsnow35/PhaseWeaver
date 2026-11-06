"""Tool abstraction: executors and the result envelope."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    KILLED = "killed"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    executor: str  # "python" | "shell" | "browser"
    timeout_s: float = 30.0
    grace_ms: int = 2000
    expected_duration_s: float | None = None
    idempotency_key: str | None = None


@dataclass
class ToolResult:
    status: ToolStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: int = 0
    truncated: bool = False
    started_at: float = 0.0
    finished_at: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK


class ToolExecutor(Protocol):
    name: str

    async def run(self, args: dict[str, Any], spec: ToolSpec) -> ToolResult: ...


def now() -> float:
    return time.monotonic()
