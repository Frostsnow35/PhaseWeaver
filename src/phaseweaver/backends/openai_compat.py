"""OpenAI-compatible inference backend.

This adapter is the proof that "engine-agnostic" is an executable claim, not a
slogan: it implements the same `InferenceBackend` protocol as the mock, so the
runtime, the lease broker and the scheduler run unchanged against a real HTTP
engine.

Capabilities it honestly does NOT have (and therefore degrades):

- no per-request KV offload -> `OFFLOAD` degrades to `DROP` (recompute).
- no server-side abort of an already-finished request -> `DROP` only forgets the
  local residency; the next turn resends the context (that is what recompute is).
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from phaseweaver.backends.base import (
    BackendCaps,
    GenStats,
    Handle,
    ParkResult,
    RestoreResult,
)
from phaseweaver.types import Disposition


@dataclass
class OpenAICompatConfig:
    base_url: str = "http://127.0.0.1:8000"
    model: str = "local-model"
    api_key: str | None = None
    timeout_s: float = 120.0
    capacity_bytes: int = 8 * 1024**3
    block_tokens: int = 16
    bytes_per_token: int = 57344
    prefill_tok_per_s: int = 3200
    decode_tok_per_s: int = 45
    #: engine-level sleep/wake is a global side effect; never enabled by default
    allow_engine_sleep: bool = False


@dataclass
class _Resident:
    tokens: int
    dropped: bool = False


class OpenAICompatBackend:
    def __init__(self, config: OpenAICompatConfig | None = None) -> None:
        self.config = config or OpenAICompatConfig()
        self._caps = BackendCaps(
            capacity_bytes=self.config.capacity_bytes,
            block_tokens=self.config.block_tokens,
            bytes_per_token=self.config.bytes_per_token,
            prefill_tok_per_s=self.config.prefill_tok_per_s,
            decode_tok_per_s=self.config.decode_tok_per_s,
            mem_bw_bps=12 * 1024**3,
            supports_retain=True,
            supports_drop=True,
            supports_offload=False,  # no per-request offload over the OpenAI API
            supports_engine_sleep=self.config.allow_engine_sleep,
        )
        self._resident: dict[str, _Resident] = {}

    # --- protocol ---------------------------------------------------------
    def caps(self) -> BackendCaps:
        return self._caps

    def available_bytes(self) -> int:
        used = sum(r.tokens for r in self._resident.values() if not r.dropped)
        return self._caps.capacity_bytes - used * self._caps.bytes_per_token

    async def submit(self, unit_id: str, context_id: str, tokens: int) -> Handle:
        self._resident[context_id] = _Resident(tokens=tokens)
        return Handle(unit_id=unit_id, context_id=context_id, resident_tokens=tokens)

    async def complete(self, handle: Handle, out_tokens: int) -> GenStats:
        start = time.monotonic()
        prompt_tokens = handle.resident_tokens
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": _dummy_prompt(prompt_tokens)}],
            "max_tokens": max(1, out_tokens),
            "temperature": 0.0,
        }
        body = await asyncio.to_thread(self._post_json, "/v1/chat/completions", payload)
        end = time.monotonic()
        usage = body.get("usage") or {}
        completion = int(usage.get("completion_tokens") or out_tokens)
        self._resident[handle.context_id] = _Resident(tokens=prompt_tokens + completion)
        handle.resident_tokens = prompt_tokens + completion
        handle.dropped = False
        return GenStats(
            prompt_tokens=int(usage.get("prompt_tokens") or prompt_tokens),
            completion_tokens=completion,
            start_ts=start,
            end_ts=end,
            ttft_ms=(end - start) * 1000,
        )

    async def abort(self, handle: Handle, reason: str) -> None:
        self._resident.pop(handle.context_id, None)
        handle.dropped = True

    async def park(self, handle: Handle, disposition: Disposition) -> ParkResult:
        start = time.monotonic()
        entry = self._resident.get(handle.context_id)
        if entry is None or disposition is Disposition.RETAIN:
            return ParkResult(Disposition.RETAIN, None, 0, 0.0)
        applied = disposition
        degraded = None
        if applied is Disposition.OFFLOAD:
            # Honest degradation: the API cannot offload a specific request.
            applied = Disposition.DROP
            degraded = Disposition.OFFLOAD
        moved = entry.tokens * self._caps.bytes_per_token
        self._resident[handle.context_id] = _Resident(tokens=entry.tokens, dropped=True)
        handle.dropped = True
        return ParkResult(
            applied=applied,
            degraded_from=degraded,
            moved_bytes=0 if applied is Disposition.DROP else moved,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def restore(self, handle: Handle) -> RestoreResult:
        start = time.monotonic()
        entry = self._resident.get(handle.context_id)
        recompute = 0
        if entry is not None and entry.dropped:
            recompute = entry.tokens
            self._resident[handle.context_id] = _Resident(tokens=entry.tokens, dropped=False)
        handle.dropped = False
        return RestoreResult(
            recompute_tokens=recompute,
            loaded_bytes=0,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def truncate(self, handle: Handle, tokens: int) -> int:
        entry = self._resident.get(handle.context_id)
        target = max(1, min(tokens, handle.resident_tokens))
        if entry is not None:
            self._resident[handle.context_id] = _Resident(tokens=target, dropped=entry.dropped)
        handle.resident_tokens = target
        return target

    # --- HTTP -------------------------------------------------------------
    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"inference request failed: {exc}") from exc
        parsed: dict[str, Any] = json.loads(raw)
        return parsed


def _dummy_prompt(tokens: int) -> str:
    """The runtime hands the adapter a token budget, not message text.

    A real deployment would pass the conversation through `runtime.reason()`;
    for protocol-level verification we synthesize a prompt of roughly `tokens`
    tokens (~4 chars each) so the server sees a realistic request size.
    """
    return "x" * max(4, tokens * 4)
