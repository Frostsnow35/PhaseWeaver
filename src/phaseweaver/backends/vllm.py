"""vLLM native admin surface — protocol-level adapter, ``[L3]-pending``.

Endpoint shapes come from the vLLM sleep-mode documentation. This module has
**not** been exercised against a real vLLM server; it makes no performance
claim and produces no ``[L3]`` result.

Precondition for every admin endpoint: the server must be started with
``VLLM_SERVER_DEV_MODE=1`` and ``--enable-sleep-mode``. Sleep / wake / reset are
ENGINE-GLOBAL side effects affecting every tenant on the process. Level 1
offloads weights to CPU and DISCARDS the entire KV cache; level 2 discards the
weights too. They therefore map to ``RELEASE`` semantics, never to a
per-context ``OFFLOAD``.

What this branch deliberately does NOT do: no ``kv_transfer_params`` injection,
no prefix-cache hit accounting, no change to ``BackendCaps`` in ``main``. The
HTTP surface cannot observe block-level refcounts, so the adapter records what
the deployment declares and what it can probe, and keeps degrading
``OFFLOAD -> DROP`` instead of pretending data was moved.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

from phaseweaver.backends.openai_compat import OpenAICompatBackend, OpenAICompatConfig

_VALID_TAGS = frozenset({"weights", "kv_cache"})


class VLLMAdminError(RuntimeError):
    """An engine-admin call was rejected, timed out, or returned an unusable body."""


@dataclass(frozen=True)
class VLLMCapabilities:
    """Probed and declared capabilities, kept separate from ``BackendCaps``.

    ``admin_endpoints`` is inferred from ``/is_sleeping`` being reachable; it is
    not proof of a particular server flag. ``prefix_caching`` / ``kv_connector``
    cannot be observed over HTTP, so they are deployment declarations.
    """

    probed: bool
    admin_endpoints: bool = False
    prefix_caching: bool = False
    kv_connector: bool = False
    kv_pool_bytes: int | None = None
    detail: str = "not probed"

    @property
    def supports_engine_sleep(self) -> bool:
        return self.probed and self.admin_endpoints

    @property
    def supports_offload(self) -> bool:
        # Declaring a connector is not the same as having an OFFLOAD data path.
        return False


class VLLMBackend(OpenAICompatBackend):
    def __init__(
        self,
        config: OpenAICompatConfig | None = None,
        *,
        prefix_caching_enabled: bool = False,
        kv_connector_enabled: bool = False,
        kv_pool_bytes: int | None = None,
        allow_collective_rpc: bool = False,
    ) -> None:
        super().__init__(config)
        self._prefix_caching_enabled = prefix_caching_enabled
        self._kv_connector_enabled = kv_connector_enabled
        self._kv_pool_bytes = kv_pool_bytes
        self._allow_collective_rpc = allow_collective_rpc
        self._vllm_caps = VLLMCapabilities(probed=False)
        # Conservative until probe(): the old adapter advertised sleep purely
        # from a config flag, which is not evidence the endpoint exists.
        self._caps = replace(
            self._caps,
            supports_engine_sleep=False,
            supports_offload=False,
            capacity_bytes=kv_pool_bytes or self._caps.capacity_bytes,
        )

    # --- capability probe -------------------------------------------------
    @property
    def capabilities(self) -> VLLMCapabilities:
        return self._vllm_caps

    def capability_snapshot(self) -> dict[str, object]:
        """Flat record for experiment reports; never a performance measurement."""
        caps = self._vllm_caps
        return {
            "engine": "vllm",
            "evidence": "[L3]-pending / protocol-level probe, no real-GPU result",
            "probed": caps.probed,
            "admin_endpoints": caps.admin_endpoints,
            "prefix_caching_declared": caps.prefix_caching,
            "kv_connector_declared": caps.kv_connector,
            "kv_pool_bytes": caps.kv_pool_bytes,
            "offload_data_path": "not implemented (OFFLOAD degrades to DROP)",
            "detail": caps.detail,
        }

    async def probe(self) -> VLLMCapabilities:
        """Probe admin endpoints once. Failure is conservative, never fatal."""
        try:
            await self._call("GET", "/is_sleeping")
        except VLLMAdminError as exc:
            caps = VLLMCapabilities(probed=False, detail=f"is_sleeping unreachable: {exc}")
        else:
            caps = VLLMCapabilities(
                probed=True,
                admin_endpoints=True,
                prefix_caching=self._prefix_caching_enabled,
                kv_connector=self._kv_connector_enabled,
                kv_pool_bytes=self._kv_pool_bytes,
                detail="/is_sleeping reachable (dev endpoints inferred)",
            )
        self._vllm_caps = caps
        self._caps = replace(
            self._caps,
            supports_engine_sleep=caps.supports_engine_sleep
            and self.config.allow_engine_sleep,
            supports_offload=caps.supports_offload,
        )
        return caps

    # --- native endpoints -------------------------------------------------
    async def is_sleeping(self) -> bool:
        return _parse_bool(await self._call("GET", "/is_sleeping"))

    async def engine_sleep(self, level: int = 1) -> None:
        self._require_sleep_opt_in()
        if level not in (1, 2):
            raise VLLMAdminError("level must be 1 or 2")
        await self._call("POST", f"/sleep?level={level}")

    async def engine_wake_up(self, tags: tuple[str, ...] | None = None) -> None:
        self._require_sleep_opt_in()
        if tags:
            invalid = set(tags) - _VALID_TAGS
            if invalid:
                raise VLLMAdminError(f"unknown wake_up tags: {sorted(invalid)}")
            path = "/wake_up?tags=" + ",".join(tags)
        else:
            path = "/wake_up"
        await self._call("POST", path)

    async def reset_prefix_cache(self) -> None:
        """Required after a level-2 wake, before serving requests again."""
        self._require_sleep_opt_in()
        await self._call("POST", "/reset_prefix_cache")

    async def reload_weights(self, weights_path: str) -> None:
        """Level-2 wake step. Disabled by default: it executes arbitrary RPC."""
        if not self._allow_collective_rpc:
            raise VLLMAdminError("collective_rpc is disabled (allow_collective_rpc=True)")
        payload: dict[str, object] = {
            "method": "reload_weights",
            "kwargs": {"weights_path": weights_path},
        }
        await self._call("POST", "/collective_rpc", payload)

    async def fetch_metrics(self) -> str:
        """Raw Prometheus text; the caller owns parsing, we do not guess names."""
        return await self._call("GET", "/metrics")

    # --- internals --------------------------------------------------------
    def _require_sleep_opt_in(self) -> None:
        if not self.config.allow_engine_sleep:
            raise VLLMAdminError("engine sleep is disabled (allow_engine_sleep=True)")

    async def _call(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> str:
        return await asyncio.to_thread(self._request, method, path, payload)

    def _request(
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
                return raw.decode("utf-8")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise VLLMAdminError(f"{method} {path} failed: {exc}") from exc


def _parse_bool(raw: str) -> bool:
    text = raw.strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VLLMAdminError(f"is_sleeping returned unusable body: {text[:80]!r}") from exc
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, dict):
        value = parsed.get("is_sleeping")
        if isinstance(value, bool):
            return value
    raise VLLMAdminError(f"is_sleeping returned unusable body: {text[:80]!r}")
