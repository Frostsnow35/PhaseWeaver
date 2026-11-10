"""SGLang native adapter — protocol-level, ``[L3]-pending``.

Endpoint shapes come from the SGLang native API documentation. This module has
**not** been exercised against a real SGLang server; it makes no performance
claim and produces no ``[L3]`` result.

The point of a native adapter is the information the OpenAI-compatible surface
does not carry: ``/server_info`` exposes the serving args and token-pool sizes,
RadixAttention owns the prefix tree, and HiCache adds L2/L3 tiers. None of those
block-level refcounts are visible over HTTP, so this adapter reports what it can
probe and what the deployment declares, and keeps ``OFFLOAD -> DROP`` rather
than pretending it moved KV.

What this branch deliberately does NOT do: no session/rid prefix-reuse data
path, no HiCache prefetch control, no prefix-hit accounting, no change to
``BackendCaps`` in ``main``.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

from phaseweaver.backends.openai_compat import OpenAICompatBackend, OpenAICompatConfig

_TOKEN_KEYS = frozenset({"max_total_num_tokens", "max_total_tokens"})


class SGLangAdminError(RuntimeError):
    """A native SGLang endpoint failed, timed out, or returned an unusable body."""


@dataclass(frozen=True)
class SGLangCapabilities:
    """Probed and declared capabilities, kept separate from ``BackendCaps``.

    ``max_total_tokens`` comes from ``/server_info`` when present. The byte
    figure derived from it is an ESTIMATE: it multiplies by the configured
    ``bytes_per_token``, not by a value reported by the engine.
    """

    probed: bool
    radix_cache: bool = False
    hicache: bool = False
    max_total_tokens: int | None = None
    kv_pool_bytes: int | None = None
    server_info_keys: tuple[str, ...] = ()
    detail: str = "not probed"

    @property
    def supports_offload(self) -> bool:
        # HiCache being enabled is not an OFFLOAD data path in this adapter.
        return False


class SGLangNativeBackend(OpenAICompatBackend):
    def __init__(
        self,
        config: OpenAICompatConfig | None = None,
        *,
        radix_cache_enabled: bool = True,
        hicache_enabled: bool = False,
        allow_flush_cache: bool = False,
    ) -> None:
        super().__init__(config)
        self._radix_cache_enabled = radix_cache_enabled
        self._hicache_enabled = hicache_enabled
        self._allow_flush_cache = allow_flush_cache
        self._sglang_caps = SGLangCapabilities(probed=False)
        self._caps = replace(self._caps, supports_offload=False)

    # --- capability probe -------------------------------------------------
    @property
    def capabilities(self) -> SGLangCapabilities:
        return self._sglang_caps

    def capability_snapshot(self) -> dict[str, object]:
        """Flat record for experiment reports; never a performance measurement."""
        caps = self._sglang_caps
        return {
            "engine": "sglang",
            "evidence": "[L3]-pending / protocol-level probe, no real-GPU result",
            "probed": caps.probed,
            "radix_cache_declared": caps.radix_cache,
            "hicache_declared": caps.hicache,
            "max_total_tokens": caps.max_total_tokens,
            "kv_pool_bytes": caps.kv_pool_bytes,
            "kv_pool_source": "max_total_tokens x configured bytes_per_token (estimate)",
            "offload_data_path": "not implemented (OFFLOAD degrades to DROP)",
            "detail": caps.detail,
        }

    async def probe(self) -> SGLangCapabilities:
        """Read /server_info once. Failure or unknown shape is conservative."""
        try:
            raw = await self._call("GET", "/server_info")
            parsed: Any = json.loads(raw)
            if not isinstance(parsed, dict):
                raise SGLangAdminError("/server_info is not a JSON object")
        except (SGLangAdminError, json.JSONDecodeError) as exc:
            caps = SGLangCapabilities(probed=False, detail=f"server_info unusable: {exc}")
        else:
            tokens = _find_token_pool(parsed)
            pool = tokens * self._caps.bytes_per_token if tokens else None
            caps = SGLangCapabilities(
                probed=True,
                radix_cache=self._radix_cache_enabled,
                hicache=self._hicache_enabled,
                max_total_tokens=tokens,
                kv_pool_bytes=pool,
                server_info_keys=tuple(sorted(str(key) for key in parsed)),
                detail=(
                    f"/server_info read; {len(parsed)} keys; "
                    + ("token pool estimated" if tokens else "no token-pool field found")
                ),
            )
        self._sglang_caps = caps
        self._caps = replace(
            self._caps,
            capacity_bytes=caps.kv_pool_bytes or self._caps.capacity_bytes,
            supports_offload=caps.supports_offload,
        )
        return caps

    # --- native endpoints -------------------------------------------------
    async def generate(self, text: str, max_new_tokens: int = 16) -> dict[str, object]:
        payload: dict[str, object] = {
            "text": text,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        }
        raw = await self._call("POST", "/generate", payload)
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SGLangAdminError(f"/generate returned unusable body: {raw[:80]!r}") from exc
        return parsed if isinstance(parsed, dict) else {"raw": parsed}

    async def flush_cache(self, timeout_s: float = 0.0) -> None:
        """Global radix-cache invalidation; explicit opt-in required."""
        if not self._allow_flush_cache:
            raise SGLangAdminError("flush_cache is disabled (allow_flush_cache=True)")
        await self._call("POST", f"/flush_cache?timeout={timeout_s:g}")

    async def get_model_info(self) -> dict[str, object]:
        raw = await self._call("GET", "/get_model_info")
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SGLangAdminError("/get_model_info returned unusable body") from exc
        return parsed if isinstance(parsed, dict) else {"raw": parsed}

    async def health_generate(self) -> str:
        return await self._call("GET", "/health_generate")

    # --- internals --------------------------------------------------------
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
            raise SGLangAdminError(f"{method} {path} failed: {exc}") from exc


def _find_token_pool(payload: dict[str, Any], depth: int = 0) -> int | None:
    """Find a positive token-pool field without assuming one server-info layout."""
    if depth > 3:
        return None
    for key, value in payload.items():
        if key in _TOKEN_KEYS and isinstance(value, int) and not isinstance(value, bool):
            return value if value > 0 else None
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_token_pool(value, depth + 1)
            if found is not None:
                return found
    return None
