"""M5: the OpenAI-compatible adapter, verified against a local fake server.

No GPU required: the point of this test is the protocol - HTTP call, capability
negotiation (OFFLOAD degrades to DROP), recompute accounting, and the runtime
running unchanged on a non-mock backend.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from phaseweaver.backends.openai_compat import OpenAICompatBackend, OpenAICompatConfig
from phaseweaver.clock import RealClock
from phaseweaver.leases import Budget
from phaseweaver.policy import PinAlwaysPolicy
from phaseweaver.runtime import AgentHandle, Runtime, RuntimeConfig
from phaseweaver.types import Disposition


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt_tokens = max(1, len(body["messages"][0]["content"]) // 4)
        completion = int(body.get("max_tokens", 1))
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion},
        }
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()


def backend(server: str) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        OpenAICompatConfig(base_url=server, capacity_bytes=64 * 1024 * 1024, timeout_s=10.0)
    )


def test_complete_reports_usage(server: str) -> None:
    async def scenario() -> None:
        b = backend(server)
        handle = await b.submit("u1", "c1", 100)
        stats = await b.complete(handle, 16)
        assert stats.completion_tokens == 16
        assert stats.prompt_tokens > 0
        assert handle.resident_tokens == stats.prompt_tokens + 16

    asyncio.run(scenario())


def test_offload_degrades_to_drop(server: str) -> None:
    async def scenario() -> None:
        b = backend(server)
        handle = await b.submit("u1", "c1", 64)
        park = await b.park(handle, Disposition.OFFLOAD)
        assert park.applied is Disposition.DROP
        assert park.degraded_from is Disposition.OFFLOAD
        restore = await b.restore(handle)
        assert restore.recompute_tokens == 64

    asyncio.run(scenario())


def test_runtime_runs_on_non_mock_backend(server: str) -> None:
    async def scenario() -> Runtime:
        b = backend(server)
        rt = Runtime(
            backend=b,
            clock=RealClock(),
            budget=Budget(total_bytes=b.caps().capacity_bytes, headroom_frac=0.0),
            policy=PinAlwaysPolicy(),
            config=RuntimeConfig(run_id="oai", simulate_tools=False),
        )
        await rt.start()
        await rt.run_task("demo", _agent, slo_s=30.0)
        await rt.shutdown()
        return rt

    rt = asyncio.run(scenario())
    assert rt.journal.by_type("phase_transitioned")
    assert not rt.broker.active_leases()


async def _agent(handle: AgentHandle) -> None:
    await handle.reason(prompt_tokens=64, out_tokens=8)
    await handle.reason(prompt_tokens=96, out_tokens=8)
