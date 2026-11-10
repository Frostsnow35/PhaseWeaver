"""SGLang native adapter: protocol-level contract tests. No GPU, no sglang package.

Evidence: [L3]-pending. These tests verify native endpoint shapes, the
/server_info token-pool mapping, conservative degradation on unknown or failed
responses, and that cancellation / errors / timeouts leave the adapter usable.
They prove nothing about real-engine behaviour.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import pytest

from phaseweaver.backends.openai_compat import OpenAICompatConfig
from phaseweaver.backends.sglang import SGLangAdminError, SGLangNativeBackend
from phaseweaver.types import Disposition

_CAPACITY = 64 * 1024 * 1024
_TOKENS = 16384


class _Handler(BaseHTTPRequestHandler):
    records: list[tuple[str, str, bytes]] = []
    mode: str = "ok"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self._handle("POST")

    def _handle(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        _Handler.records.append((method, self.path, body))
        if _Handler.mode == "fail":
            self._finish(b"boom", 500)
            return
        if _Handler.mode == "slow":
            time.sleep(0.3)
        path = urlparse(self.path).path
        if path == "/server_info":
            payload: object = (
                {} if _Handler.mode == "unknown_shape" else {"max_total_num_tokens": _TOKENS}
            )
            self._finish(json.dumps(payload).encode("utf-8"))
        elif path == "/get_model_info":
            self._finish(b'{"model_path":"qwen2.5-0.5b","is_generation":true}')
        elif path == "/health_generate":
            self._finish(b"ok")
        elif path == "/generate":
            self._finish(b'{"text":"reply","meta_info":{"completion_tokens":4}}')
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


def backend(server: str, **kwargs: Any) -> SGLangNativeBackend:
    config = OpenAICompatConfig(
        base_url=server,
        timeout_s=kwargs.pop("timeout_s", 5.0),
        capacity_bytes=_CAPACITY,
    )
    return SGLangNativeBackend(config, **kwargs)


def test_server_info_probe_marks_pool_as_estimate(server: str) -> None:
    async def scenario() -> tuple[int, dict[str, object]]:
        b = backend(server, radix_cache_enabled=True, hicache_enabled=True,
                    engine_version="v0.5.0")
        caps = await b.probe()
        assert caps.max_total_tokens == _TOKENS
        assert caps.supports_offload is False
        assert "no token-pool" not in caps.detail
        return b.caps().capacity_bytes, b.capability_snapshot()

    capacity, snapshot = asyncio.run(scenario())
    assert snapshot["kv_pool_bytes"] == _TOKENS * 57344
    assert snapshot["kv_pool_bytes_origin"] == "estimated"
    assert snapshot["max_total_tokens_origin"] == "probed"
    assert snapshot["radix_cache_declared_origin"] == "declared"
    assert snapshot["contract_version"] == "sglang-native/1"
    assert snapshot["engine_version_checked"] == "v0.5.0"
    assert snapshot["probed"] is True
    assert snapshot["radix_cache_declared"] is True
    assert snapshot["hicache_declared"] is True
    assert str(snapshot["evidence"]).startswith("[L3]-pending")
    assert ("GET", "/server_info", b"") in _Handler.records
    # An estimate derived from an assumed bytes_per_token must NOT become the
    # lease budget without an explicit opt-in.
    assert capacity == _CAPACITY


def test_estimated_capacity_needs_explicit_opt_in(server: str) -> None:
    async def scenario() -> int:
        b = backend(server, accept_estimated_capacity=True)
        await b.probe()
        return b.caps().capacity_bytes

    assert asyncio.run(scenario()) == _TOKENS * 57344


def test_probe_unknown_shape_stays_conservative(server: str) -> None:
    _Handler.mode = "unknown_shape"

    async def scenario() -> tuple[int, bool]:
        b = backend(server)
        caps = await b.probe()
        assert caps.probed is True
        assert caps.max_total_tokens is None
        assert "no token-pool" in caps.detail
        return b.caps().capacity_bytes, b.caps().supports_offload

    assert asyncio.run(scenario()) == (_CAPACITY, False)


def test_probe_failure_stays_conservative(server: str) -> None:
    _Handler.mode = "fail"

    async def scenario() -> dict[str, object]:
        b = backend(server)
        caps = await b.probe()
        assert caps.probed is False
        assert b.caps().capacity_bytes == _CAPACITY
        assert b.caps().supports_offload is False
        return b.capability_snapshot()

    snapshot = asyncio.run(scenario())
    assert snapshot["probed"] is False
    assert snapshot["offload_data_path"].startswith("not implemented")


def test_generate_uses_native_path_and_text_field(server: str) -> None:
    async def scenario() -> dict[str, object]:
        b = backend(server)
        return await b.generate("hello", max_new_tokens=8)

    result = asyncio.run(scenario())
    assert result["text"] == "reply"
    method, path, body = _Handler.records[0]
    assert (method, path) == ("POST", "/generate")
    assert json.loads(body)["text"] == "hello"
    assert json.loads(body)["sampling_params"]["max_new_tokens"] == 8


def test_flush_cache_requires_opt_in_and_encodes_timeout(server: str) -> None:
    async def scenario() -> None:
        b = backend(server)
        with pytest.raises(SGLangAdminError, match="disabled"):
            await b.flush_cache()
        allowed = backend(server, allow_flush_cache=True)
        await allowed.flush_cache(30.0)

    asyncio.run(scenario())
    assert ("POST", "/flush_cache?timeout=30", b"") in _Handler.records


def test_model_info_and_health_paths(server: str) -> None:
    async def scenario() -> tuple[dict[str, object], str]:
        b = backend(server)
        return await b.get_model_info(), await b.health_generate()

    info, health = asyncio.run(scenario())
    assert info["model_path"] == "qwen2.5-0.5b"
    assert health == "ok"
    assert [path for _, path, _ in _Handler.records] == ["/get_model_info", "/health_generate"]


def test_park_degrades_offload_even_with_hicache(server: str) -> None:
    async def scenario() -> object:
        b = backend(server, hicache_enabled=True)
        await b.probe()
        handle = await b.submit("u1", "c1", 128)
        result = await b.park(handle, Disposition.OFFLOAD)
        return result.applied, result.degraded_from

    assert asyncio.run(scenario()) == (Disposition.DROP, Disposition.OFFLOAD)


def test_timeout_does_not_poison_state(server: str) -> None:
    async def scenario() -> None:
        b = backend(server, timeout_s=0.05)
        _Handler.mode = "slow"
        caps = await b.probe()
        assert caps.probed is False
        _Handler.mode = "ok"
        assert b.caps().capacity_bytes == _CAPACITY
        await b.health_generate()

    asyncio.run(scenario())
    assert ("GET", "/health_generate", b"") in _Handler.records


def test_native_call_error_is_wrapped_not_leaked(server: str) -> None:
    _Handler.mode = "fail"

    async def scenario() -> None:
        b = backend(server)
        with pytest.raises(SGLangAdminError, match="/generate failed"):
            await b.generate("hello")

    asyncio.run(scenario())


def test_cancel_does_not_poison_state(server: str) -> None:
    async def scenario() -> None:
        b = backend(server)
        task = asyncio.create_task(b.probe())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert b.caps().supports_offload is False
        await b.health_generate()

    asyncio.run(scenario())
    assert ("GET", "/health_generate", b"") in _Handler.records
