"""Probe a SGLang server's native surface. ``[L3]-pending``.

Start the server first::

    python3 -m sglang.launch_server --model-path <model> --port 30000

Then run ``python examples/sglang_probe.py``. This example only reads
``/server_info`` and ``/get_model_info``. It deliberately does NOT call
``/flush_cache`` (global radix-cache invalidation) or ``/generate``; those
change server state or start work, and are covered by contract tests instead.
"""

from __future__ import annotations

import asyncio
import json

from phaseweaver.backends.openai_compat import OpenAICompatConfig
from phaseweaver.backends.sglang import SGLangNativeBackend


async def main() -> None:
    backend = SGLangNativeBackend(
        OpenAICompatConfig(base_url="http://127.0.0.1:30000"),
        radix_cache_enabled=True,
        hicache_enabled=False,
        # Set to the exact SGLang release once endpoint shapes are verified on hardware.
        engine_version="unverified",
    )
    await backend.probe()
    print(json.dumps(backend.capability_snapshot(), indent=2, ensure_ascii=False))
    info = await backend.get_model_info()
    print("model:", info.get("model_path", "unknown"))


if __name__ == "__main__":
    asyncio.run(main())
