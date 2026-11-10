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

Transport and capability reporting come from ``backends/http_admin.py`` and
``backends/capability.py``. The KV-pool byte figure is an ESTIMATE
(``max_total_tokens x configured bytes_per_token``) and never silently becomes
the lease budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from phaseweaver.backends.capability import (
    CapabilityField,
    CapabilityOrigin,
    CapabilityReport,
)
from phaseweaver.backends.http_admin import (
    AdminHttpClient,
    AdminHttpConfig,
    BackendAdminError,
)
from phaseweaver.backends.openai_compat import OpenAICompatBackend, OpenAICompatConfig

_TOKEN_KEYS = frozenset({"max_total_num_tokens", "max_total_tokens"})
_CONTRACT_VERSION = "sglang-native/1"
_EVIDENCE = "[L3]-pending / protocol-level probe, no real-GPU result"


class SGLangAdminError(BackendAdminError):
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
        accept_estimated_capacity: bool = False,
        engine_version: str = "unverified",
    ) -> None:
        super().__init__(config)
        self._radix_cache_enabled = radix_cache_enabled
        self._hicache_enabled = hicache_enabled
        self._allow_flush_cache = allow_flush_cache
        self._accept_estimated_capacity = accept_estimated_capacity
        self._engine_version = engine_version
        self._http = AdminHttpClient(
            AdminHttpConfig(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout_s=self.config.timeout_s,
            )
        )
        self._sglang_caps = SGLangCapabilities(probed=False)
        self._report = self._build_report(self._sglang_caps)
        self._caps = replace(self._caps, supports_offload=False)

    # --- capability probe -------------------------------------------------
    @property
    def capabilities(self) -> SGLangCapabilities:
        return self._sglang_caps

    @property
    def capability_report(self) -> CapabilityReport:
        return self._report

    def capability_snapshot(self) -> dict[str, object]:
        """Flat record for experiment reports; never a performance measurement."""
        return self._report.as_dict()

    def _build_report(self, caps: SGLangCapabilities) -> CapabilityReport:
        declared = CapabilityOrigin.DECLARED
        unknown = CapabilityOrigin.UNKNOWN
        fields = (
            CapabilityField("radix_cache_declared", caps.radix_cache, declared,
                            "server radix-cache flag; refcounts stay engine-internal"),
            CapabilityField("hicache_declared", caps.hicache, declared,
                            "server --enable-hierarchical-cache"),
            CapabilityField(
                "max_total_tokens", caps.max_total_tokens,
                CapabilityOrigin.PROBED if caps.max_total_tokens else unknown,
                "read from GET /server_info",
            ),
            CapabilityField(
                "kv_pool_bytes", caps.kv_pool_bytes,
                CapabilityOrigin.ESTIMATED if caps.kv_pool_bytes else unknown,
                "max_total_tokens x configured bytes_per_token (not engine-reported)",
            ),
            CapabilityField("kv_pool_source",
                            "max_total_tokens x configured bytes_per_token (estimate)",
                            declared),
            CapabilityField("offload_data_path",
                            "not implemented (OFFLOAD degrades to DROP)", declared),
            CapabilityField("supports_offload", caps.supports_offload, declared),
            CapabilityField("server_info_keys", caps.server_info_keys,
                            CapabilityOrigin.PROBED if caps.probed else unknown),
        )
        return CapabilityReport(
            engine="sglang",
            contract_version=_CONTRACT_VERSION,
            evidence=_EVIDENCE,
            probed=caps.probed,
            fields=fields,
            engine_version_checked=self._engine_version,
            detail=caps.detail,
        )

    async def probe(self) -> SGLangCapabilities:
        """Read /server_info once. Failure or unknown shape is conservative."""
        try:
            raw = await self._call("GET", "/server_info")
            parsed: Any = json.loads(raw)
            if not isinstance(parsed, dict):
                raise SGLangAdminError("/server_info is not a JSON object")
        except (BackendAdminError, json.JSONDecodeError) as exc:
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
        self._report = self._build_report(caps)
        # The estimate only enters the lease budget on explicit opt-in.
        capacity = (
            caps.kv_pool_bytes
            if self._accept_estimated_capacity and caps.kv_pool_bytes
            else self._caps.capacity_bytes
        )
        self._caps = replace(
            self._caps,
            capacity_bytes=capacity,
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
        try:
            return await self._http.call(method, path, payload)
        except BackendAdminError as exc:
            raise SGLangAdminError(str(exc)) from exc


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
