"""Engine-agnostic HTTP admin primitives and a self-describing capability report.

Adding a new inference engine should mean writing an endpoint map and a probe,
not copying another ``urllib`` boilerplate or inventing another snapshot dict.
This module contains no engine-specific path, flag or version knowledge.

``BackendAdminError`` deliberately derives from ``RuntimeError``: the existing
OpenAI-compatible adapter already raised ``RuntimeError`` on transport failure,
so callers catching that type keep working.

Cancellation caveat: ``call()`` uses ``asyncio.to_thread``. Cancelling the
awaiting coroutine does NOT stop the in-flight HTTP request; a global-side-effect
endpoint (sleep / flush / reset) may still complete on the server. Adapters must
re-read engine state after a cancelled call instead of assuming nothing happened.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class BackendAdminError(RuntimeError):
    """An engine management call failed, timed out, or returned an unusable body."""


@dataclass(frozen=True)
class AdminHttpConfig:
    base_url: str
    api_key: str | None = None
    timeout_s: float = 120.0


class AdminHttpClient:
    """One transport policy for every engine adapter: auth, timeout, error wrapping."""

    def __init__(self, config: AdminHttpConfig) -> None:
        self.config = config

    def request_text(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> str:
        url = self.config.base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as resp:
                raw: bytes = resp.read()
        except urllib.error.HTTPError as exc:
            raise BackendAdminError(f"{method} {path} failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise BackendAdminError(f"{method} {path} failed: {exc}") from exc
        return raw.decode("utf-8")

    def request_json(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> Any:
        raw = self.request_text(method, path, payload)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackendAdminError(
                f"{method} {path} returned invalid JSON: {raw[:80]!r}"
            ) from exc

    async def call(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> str:
        """Offload the blocking request to a worker thread; see cancellation caveat."""
        return await asyncio.to_thread(self.request_text, method, path, payload)
