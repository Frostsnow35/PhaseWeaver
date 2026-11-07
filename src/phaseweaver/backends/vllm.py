"""vLLM backend: OpenAI-compatible data path plus optional engine admin calls.

The admin surface (`/sleep?level=1`, `/wake_up`) is an **engine-global** side
effect: it parks every tenant on that process. It therefore sits behind an
explicit opt-in and must be disclosed in any experiment report.
"""

from __future__ import annotations

import asyncio
import urllib.request
from dataclasses import replace

from phaseweaver.backends.openai_compat import OpenAICompatBackend, OpenAICompatConfig


class VLLMBackend(OpenAICompatBackend):
    def __init__(self, config: OpenAICompatConfig | None = None) -> None:
        super().__init__(config)
        # vLLM can offload KV through a connector, but only when the deployment is
        # configured with one; absent that we keep advertising "no offload" so the
        # runtime degrades to DROP instead of pretending.
        self._caps = replace(
            self._caps,
            supports_engine_sleep=bool(self.config.allow_engine_sleep),
        )

    async def engine_sleep(self, level: int = 1) -> None:
        if not self.config.allow_engine_sleep:
            raise RuntimeError("engine sleep is disabled (set allow_engine_sleep=True)")
        await asyncio.to_thread(self._get, f"/sleep?level={level}")

    async def engine_wake_up(self) -> None:
        if not self.config.allow_engine_sleep:
            raise RuntimeError("engine wake_up is disabled (set allow_engine_sleep=True)")
        await asyncio.to_thread(self._get, "/wake_up")

    def _get(self, path: str) -> None:
        url = self.config.base_url.rstrip("/") + path
        with urllib.request.urlopen(url, timeout=self.config.timeout_s) as resp:
            resp.read()
