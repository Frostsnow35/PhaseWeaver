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

Transport and capability reporting come from the engine-agnostic modules in
``backends/http_admin.py`` and ``backends/capability.py``. Adding another engine
should reuse them, not copy this file's HTTP plumbing.
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

_VALID_TAGS = frozenset({"weights", "kv_cache"})
_CONTRACT_VERSION = "vllm-admin/1"
_EVIDENCE = "[L3]-pending / protocol-level probe, no real-GPU result"


class VLLMAdminError(BackendAdminError):
    """An engine-admin call was rejected, timed out, or returned an unusable body."""


@dataclass(frozen=True)
class VLLMCapabilities:
    """Probed and declared capabilities, kept separate from ``BackendCaps``.

    ``admin_endpoints`` is inferred from a *valid* ``/is_sleeping`` response; it
    is not proof of a particular server flag. ``prefix_caching`` / ``kv_connector``
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
        accept_declared_capacity: bool = False,
        engine_version: str = "unverified",
    ) -> None:
        super().__init__(config)
        self._prefix_caching_enabled = prefix_caching_enabled
        self._kv_connector_enabled = kv_connector_enabled
        self._kv_pool_bytes = kv_pool_bytes
        self._allow_collective_rpc = allow_collective_rpc
        self._accept_declared_capacity = accept_declared_capacity
        self._engine_version = engine_version
        self._http = AdminHttpClient(
            AdminHttpConfig(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout_s=self.config.timeout_s,
            )
        )
        self._vllm_caps = VLLMCapabilities(probed=False)
        self._report = self._build_report(self._vllm_caps)
        # Conservative until probe(): the old adapter advertised sleep purely
        # from a config flag, which is not evidence the endpoint exists.
        self._caps = replace(
            self._caps,
            supports_engine_sleep=False,
            supports_offload=False,
            capacity_bytes=self._initial_capacity(),
        )

    # --- capability probe -------------------------------------------------
    @property
    def capabilities(self) -> VLLMCapabilities:
        return self._vllm_caps

    @property
    def capability_report(self) -> CapabilityReport:
        return self._report

    def capability_snapshot(self) -> dict[str, object]:
        """Flat record for experiment reports; never a performance measurement."""
        return self._report.as_dict()

    def _initial_capacity(self) -> int:
        # A declared KV-pool size is NOT an observation. It only enters the lease
        # budget when the caller explicitly opts in.
        if self._accept_declared_capacity and self._kv_pool_bytes:
            return self._kv_pool_bytes
        return self._caps.capacity_bytes

    def _build_report(self, caps: VLLMCapabilities) -> CapabilityReport:
        declared = CapabilityOrigin.DECLARED
        unknown = CapabilityOrigin.UNKNOWN
        fields = (
            CapabilityField(
                "admin_endpoints",
                caps.admin_endpoints,
                CapabilityOrigin.PROBED if caps.probed else unknown,
                "inferred from GET /is_sleeping; does not prove POST /sleep is enabled",
            ),
            CapabilityField("prefix_caching_declared", caps.prefix_caching, declared,
                            "server flag --enable-prefix-caching"),
            CapabilityField("kv_connector_declared", caps.kv_connector, declared,
                            "server --kv-transfer-config"),
            CapabilityField(
                "kv_pool_bytes", caps.kv_pool_bytes,
                declared if caps.kv_pool_bytes else unknown,
                "deployment declaration, not measured",
            ),
            CapabilityField("offload_data_path",
                            "not implemented (OFFLOAD degrades to DROP)", declared),
            CapabilityField("supports_offload", caps.supports_offload, declared),
            CapabilityField("supports_engine_sleep", caps.supports_engine_sleep, declared,
                            "also requires allow_engine_sleep opt-in"),
        )
        return CapabilityReport(
            engine="vllm",
            contract_version=_CONTRACT_VERSION,
            evidence=_EVIDENCE,
            probed=caps.probed,
            fields=fields,
            engine_version_checked=self._engine_version,
            detail=caps.detail,
        )

    async def probe(self) -> VLLMCapabilities:
        """Probe admin endpoints once. Failure is conservative, never fatal."""
        try:
            raw = await self._call("GET", "/is_sleeping")
            # A 200 with an empty or HTML body is not evidence of a dev endpoint.
            _parse_bool(raw)
        except BackendAdminError as exc:
            caps = VLLMCapabilities(probed=False, detail=f"is_sleeping unusable: {exc}")
        else:
            caps = VLLMCapabilities(
                probed=True,
                admin_endpoints=True,
                prefix_caching=self._prefix_caching_enabled,
                kv_connector=self._kv_connector_enabled,
                kv_pool_bytes=self._kv_pool_bytes,
                detail="/is_sleeping returned a valid body (dev endpoints inferred)",
            )
        self._vllm_caps = caps
        self._report = self._build_report(caps)
        self._caps = replace(
            self._caps,
            supports_engine_sleep=caps.supports_engine_sleep
            and self.config.allow_engine_sleep,
            supports_offload=caps.supports_offload,
            capacity_bytes=self._initial_capacity(),
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
        try:
            return await self._http.call(method, path, payload)
        except BackendAdminError as exc:
            raise VLLMAdminError(str(exc)) from exc


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
