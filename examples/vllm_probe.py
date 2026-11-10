"""Probe a vLLM dev-mode server's admin surface. ``[L3]-pending``.

Server must be started with ``VLLM_SERVER_DEV_MODE=1 vllm serve <model>
--enable-sleep-mode``. Sleep/wake are engine-global side effects: do not run
this against a shared instance. No GPU is needed to see the probe result, but
the calls are only meaningful against a real engine, which this repository has
not verified.

    python examples/vllm_probe.py
"""

from __future__ import annotations

import asyncio
import json

from phaseweaver.backends.openai_compat import OpenAICompatConfig
from phaseweaver.backends.vllm import VLLMBackend


async def main() -> None:
    backend = VLLMBackend(
        OpenAICompatConfig(base_url="http://127.0.0.1:8000", allow_engine_sleep=True),
        prefix_caching_enabled=True,
        kv_connector_enabled=False,
        # Set to the exact vLLM release once endpoint shapes are verified on hardware.
        engine_version="unverified",
    )
    caps = await backend.probe()
    print(json.dumps(backend.capability_snapshot(), indent=2, ensure_ascii=False))
    if not caps.supports_engine_sleep:
        print("admin endpoints unavailable; stop here, do not call sleep/wake")
        return
    await backend.engine_sleep(1)
    await backend.engine_wake_up()


if __name__ == "__main__":
    asyncio.run(main())
