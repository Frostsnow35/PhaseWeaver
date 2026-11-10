"""vLLM native adapter: protocol-level contract tests. No GPU, no vLLM package.

Evidence: [L3]-pending. These tests verify HTTP method / path / query shapes,
conservative capability degradation, and that cancellation / errors / timeouts
leave the adapter usable. They prove nothing about real-engine behaviour.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import pytest

from phaseweaver.backends.openai_compat import OpenAICompatConfig
from phaseweaver.backends.vllm import VLLMAdminError, VLLMBackend
from phaseweaver.types import Disposition


class _Handler(BaseHTTPRequestHandler):
    records: list[tuple[str, str]] = []
    mode: str = "ok"
    sleep_body: str = "false"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self._handle("POST")

    def _handle(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        _Handler.records.append((method, self.path))
        if _Handler.mode == "fail":
            self._finish(b"boom", 500)
            return
        if _Handler.mode == "slow":
            time.sleep(0.3)
        path = urlparse(self.path).path
        if path == "/is_sleeping":
            self._finish(_Handler.sleep_body.encode("utf-8"))
        elif path == "/metrics":
            self._finish(b"# TYPE phaseweaver_probe gauge\n")
        else:
            self._finish(b"")

    def _finish(self, body: bytes, status: int = 200) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

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


@pytest.fixture(autouse=True)
def _reset(server: str) -> None:
    _Handler.records.clear()
    _Handler.mode = "ok"
    _Handler.sleep_body = "false"


def backend(server: str, **kwargs: Any) -> VLLMBackend:
    config = OpenAICompatConfig(
        base_url=server, timeout_s=kwargs.pop("timeout_s", 5.0), allow_engine_sleep=True
    )
    return VLLMBackend(config, **kwargs)


def test_sleep_uses_post_and_level(server: str) -> None:
    async def scenario() -> None:
        b = backend(server)
        await b.engine_sleep(1)
        await b.engine_sleep(2)

    asyncio.run(scenario())
    assert ("POST", "/sleep?level=1") in _Handler.records
    assert ("POST", "/sleep?level=2") in _Handler.records
    assert all(method == "POST" for method, _ in _Handler.records)


def test_level2_wake_sequence_uses_tags_and_reset(server: str) -> None:
    async def scenario() -> None:
        b = backend(server, allow_collective_rpc=True)
        await b.engine_sleep(2)
        await b.engine_wake_up(("weights",))
        await b.reload_weights("Qwen/Qwen2.5-0.5B-Instruct")
        await b.engine_wake_up(("kv_cache",))
        await b.reset_prefix_cache()

    asyncio.run(scenario())
    paths = [path for _, path in _Handler.records]
    assert paths == [
        "/sleep?level=2",
        "/wake_up?tags=weights",
        "/collective_rpc",
        "/wake_up?tags=kv_cache",
        "/reset_prefix_cache",
    ]


def test_is_sleeping_parses_plain_and_json(server: str) -> None:
    async def scenario() -> list[bool]:
        b = backend(server)
        out = [await b.is_sleeping()]
        _Handler.sleep_body = '{"is_sleeping": true}'
        out.append(await b.is_sleeping())
        return out

    assert asyncio.run(scenario()) == [False, True]


def test_probe_records_origins_and_never_fakes_offload(server: str) -> None:
    async def scenario() -> tuple[dict[str, object], bool, int]:
        b = backend(server, prefix_caching_enabled=True, kv_connector_enabled=True,
                    kv_pool_bytes=4 * 1024**3)
        caps = await b.probe()
        return b.capability_snapshot(), caps.supports_offload, b.caps().capacity_bytes

    snapshot, offload, capacity = asyncio.run(scenario())
    assert snapshot["probed"] is True
    assert snapshot["admin_endpoints"] is True
    assert snapshot["admin_endpoints_origin"] == "probed"
    assert snapshot["prefix_caching_declared"] is True
    assert snapshot["prefix_caching_declared_origin"] == "declared"
    assert snapshot["kv_connector_declared"] is True
    assert snapshot["kv_pool_bytes"] == 4 * 1024**3
    assert snapshot["kv_pool_bytes_origin"] == "declared"
    assert snapshot["contract_version"] == "vllm-admin/1"
    assert snapshot["engine_version_checked"] == "unverified"
    assert str(snapshot["offload_data_path"]).startswith("not implemented")
    assert str(snapshot["evidence"]).startswith("[L3]-pending")
    assert offload is False
    # A declared pool size must NOT silently become the lease budget.
    assert capacity == 8 * 1024**3


def test_declared_capacity_needs_explicit_opt_in(server: str) -> None:
    async def scenario() -> int:
        b = backend(server, kv_pool_bytes=4 * 1024**3, accept_declared_capacity=True,
                    engine_version="v0.11.0")
        await b.probe()
        return b.caps().capacity_bytes

    assert asyncio.run(scenario()) == 4 * 1024**3


def test_probe_rejects_empty_200_body(server: str) -> None:
    _Handler.sleep_body = ""

    async def scenario() -> tuple[bool, str]:
        b = backend(server)
        caps = await b.probe()
        return caps.probed, caps.detail

    probed, detail = asyncio.run(scenario())
    assert probed is False
    assert "is_sleeping unusable" in detail


def test_probe_failure_is_conservative(server: str) -> None:
    _Handler.mode = "fail"

    async def scenario() -> VLLMBackend:
        b = backend(server)
        caps = await b.probe()
        assert caps.probed is False
        assert caps.supports_engine_sleep is False
        assert b.caps().supports_offload is False
        assert b.caps().supports_engine_sleep is False
        return b

    b = asyncio.run(scenario())
    assert b.capability_snapshot()["probed"] is False


def test_park_still_degrades_offload_to_drop(server: str) -> None:
    async def scenario() -> object:
        b = backend(server, kv_connector_enabled=True)
        await b.probe()
        handle = await b.submit("u1", "c1", 128)
        result = await b.park(handle, Disposition.OFFLOAD)
        return result.applied, result.degraded_from

    assert asyncio.run(scenario()) == (Disposition.DROP, Disposition.OFFLOAD)


def test_timeout_does_not_poison_state(server: str) -> None:
    async def scenario() -> None:
        b = backend(server, timeout_s=0.05)
        _Handler.mode = "slow"
        with pytest.raises(VLLMAdminError):
            await b.engine_sleep(1)
        _Handler.mode = "ok"
        assert b.caps().supports_engine_sleep is False
        await b.engine_wake_up()

    asyncio.run(scenario())
    assert ("POST", "/wake_up") in _Handler.records


def test_cancel_does_not_poison_state(server: str) -> None:
    async def scenario() -> None:
        b = backend(server)
        task = asyncio.create_task(b.engine_sleep(1))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await b.engine_wake_up()
        assert b.caps().supports_offload is False

    asyncio.run(scenario())
    assert ("POST", "/wake_up") in _Handler.records


def test_admin_calls_require_opt_in(server: str) -> None:
    async def scenario() -> None:
        config = OpenAICompatConfig(base_url=server, timeout_s=5.0)
        b = VLLMBackend(config)
        with pytest.raises(VLLMAdminError, match="disabled"):
            await b.engine_sleep(1)
        with pytest.raises(VLLMAdminError, match="disabled"):
            await b.engine_wake_up()
        with pytest.raises(VLLMAdminError, match="disabled"):
            await b.reload_weights("model")

    asyncio.run(scenario())
    assert _Handler.records == []


def test_invalid_level_and_tags_are_rejected_locally(server: str) -> None:
    async def scenario() -> None:
        b = backend(server)
        with pytest.raises(VLLMAdminError, match="level"):
            await b.engine_sleep(3)
        with pytest.raises(VLLMAdminError, match="tags"):
            await b.engine_wake_up(("bogus",))

    asyncio.run(scenario())
    assert _Handler.records == []
