"""Contract tests for the engine-agnostic admin HTTP client and capability schema.

No GPU, no engine package. These tests pin the transport policy (auth, JSON,
status codes, timeouts) and the rule that a capability value always exposes its
origin, so two adapters cannot silently publish same-key/different-evidence data.
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

from phaseweaver.backends.base import BackendCaps
from phaseweaver.backends.capability import (
    CapabilityField,
    CapabilityOrigin,
    CapabilityReport,
    caps_snapshot,
)
from phaseweaver.backends.http_admin import (
    AdminHttpClient,
    AdminHttpConfig,
    BackendAdminError,
)


class _Handler(BaseHTTPRequestHandler):
    records: list[tuple[str, str, bytes, str]] = []
    mode: str = "ok"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self._handle("POST")

    def _handle(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        _Handler.records.append(
            (method, self.path, body, self.headers.get("Authorization", ""))
        )
        if _Handler.mode == "slow":
            time.sleep(0.3)
        path = urlparse(self.path).path
        if path == "/error":
            self._finish(b"engine rejected", 500)
        elif path == "/bad-json":
            self._finish(b"not json")
        elif path == "/json":
            self._finish(b'{"ready": true}')
        elif path == "/json-list":
            self._finish(b"[1, 2]")
        else:
            self._finish(b"plain ok")

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


def client(server: str, **kwargs: Any) -> AdminHttpClient:
    return AdminHttpClient(AdminHttpConfig(base_url=server, **kwargs))


def test_request_text_posts_json_and_auth_header(server: str) -> None:
    text = client(server, api_key="secret").request_text(
        "POST", "/generate", {"text": "hi"}
    )
    assert text == "plain ok"
    method, path, body, auth = _Handler.records[0]
    assert (method, path) == ("POST", "/generate")
    assert json.loads(body) == {"text": "hi"}
    assert auth == "Bearer secret"


def test_request_json_returns_any_json_and_guard_caller_type(server: str) -> None:
    c = client(server)
    assert c.request_json("GET", "/json") == {"ready": True}
    assert c.request_json("GET", "/json-list") == [1, 2]


def test_http_error_carries_status(server: str) -> None:
    with pytest.raises(BackendAdminError, match="HTTP 500"):
        client(server).request_text("POST", "/error")


def test_invalid_json_is_wrapped(server: str) -> None:
    with pytest.raises(BackendAdminError, match="invalid JSON"):
        client(server).request_json("GET", "/bad-json")


def test_timeout_is_wrapped_and_client_stays_usable(server: str) -> None:
    c = client(server, timeout_s=0.05)
    _Handler.mode = "slow"
    with pytest.raises(BackendAdminError, match="failed"):
        c.request_text("GET", "/plain")
    _Handler.mode = "ok"
    assert c.request_text("GET", "/plain") == "plain ok"


def test_async_call_uses_transport(server: str) -> None:
    async def scenario() -> str:
        return await client(server).call("GET", "/json")

    assert asyncio.run(scenario()) == '{"ready": true}'


def test_capability_report_exposes_origin_for_every_value() -> None:
    report = CapabilityReport(
        engine="example",
        contract_version="example/1",
        evidence="[L3]-pending",
        probed=True,
        fields=(
            CapabilityField("kv_pool_bytes", 4096, CapabilityOrigin.ESTIMATED,
                            "tokens x assumed bytes_per_token"),
            CapabilityField("admin_endpoints", True, CapabilityOrigin.PROBED),
        ),
    )
    flat = report.as_dict()
    assert flat["kv_pool_bytes"] == 4096
    assert flat["kv_pool_bytes_origin"] == "estimated"
    assert str(flat["kv_pool_bytes_detail"]).startswith("tokens x")
    assert flat["admin_endpoints_origin"] == "probed"
    assert flat["engine_version_checked"] == "unverified"
    assert report.field("kv_pool_bytes") is not None
    assert report.field("missing") is None


def test_caps_snapshot_never_claims_measurement() -> None:
    caps = BackendCaps(
        capacity_bytes=1024,
        block_tokens=16,
        bytes_per_token=57344,
        prefill_tok_per_s=3200,
        decode_tok_per_s=45,
        mem_bw_bps=12 * 1024**3,
    )
    flat = caps_snapshot(caps, engine="mock").as_dict()
    assert flat["probed"] is False
    assert flat["capacity_bytes"] == 1024
    assert flat["capacity_bytes_origin"] == "declared"
    assert all(
        value != "probed"
        for key, value in flat.items()
        if key.endswith("_origin")
    )
