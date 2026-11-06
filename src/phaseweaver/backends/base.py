"""Inference backend protocol: the only contract between PhaseWeaver and an engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from phaseweaver.types import ContextId, Disposition


@dataclass(frozen=True)
class BackendCaps:
    capacity_bytes: int
    block_tokens: int
    bytes_per_token: int
    prefill_tok_per_s: int
    decode_tok_per_s: int
    mem_bw_bps: int
    supports_retain: bool = True
    supports_drop: bool = True
    supports_offload: bool = False
    supports_engine_sleep: bool = False
    #: batching efficiency: per-stream rate = base / (1 + alpha*(n-1)).
    #: 0.0 means "no compute contention" (each stream is alone).
    contention_alpha: float = 0.15
    #: integration step of the shared-throughput simulation, in virtual seconds
    time_slice_s: float = 0.05

    @property
    def blocks_total(self) -> int:
        block_bytes = self.block_tokens * self.bytes_per_token
        return max(1, self.capacity_bytes // block_bytes)


class EngineEventKind(StrEnum):
    BLOCK_STORED = "block_stored"
    BLOCK_REMOVED = "block_removed"
    REQUEST_PREEMPTED = "request_preempted"
    QUEUE_DEPTH = "queue_depth"


@dataclass(frozen=True)
class EngineEvent:
    kind: EngineEventKind
    context_id: ContextId | None
    blocks: int = 0
    payload: dict[str, object] | None = None


@dataclass
class Handle:
    unit_id: str
    context_id: ContextId
    resident_tokens: int = 0
    dropped: bool = False


@dataclass(frozen=True)
class GenStats:
    prompt_tokens: int
    completion_tokens: int
    start_ts: float
    end_ts: float
    ttft_ms: float
    recompute_tokens: int = 0


@dataclass(frozen=True)
class ParkResult:
    applied: Disposition
    degraded_from: Disposition | None
    moved_bytes: int
    latency_ms: float


@dataclass(frozen=True)
class RestoreResult:
    recompute_tokens: int
    loaded_bytes: int
    latency_ms: float


class InferenceBackend(Protocol):
    def caps(self) -> BackendCaps: ...

    def available_bytes(self) -> int: ...

    async def submit(self, unit_id: str, context_id: str, tokens: int) -> Handle: ...

    async def complete(self, handle: Handle, out_tokens: int) -> GenStats: ...

    async def abort(self, handle: Handle, reason: str) -> None: ...

    async def park(self, handle: Handle, disposition: Disposition) -> ParkResult: ...

    async def restore(self, handle: Handle) -> RestoreResult: ...

    async def truncate(self, handle: Handle, tokens: int) -> int:
        """Shrink a context to at most `tokens`; returns the resulting size (K11)."""
        ...
