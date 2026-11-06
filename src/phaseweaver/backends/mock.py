"""Deterministic mock inference backend.

- time comes only from the injected Clock (no real sleeping)
- KV is allocated in blocks; running out raises EngineOutOfMemory, which means
  the runtime's lease accounting is wrong
- engine events use the same vocabulary as vLLM kv_events so the runtime takes
  one single perception path (no mock-only shortcuts)
"""

from __future__ import annotations

from dataclasses import dataclass

from phaseweaver.backends.base import (
    BackendCaps,
    EngineEvent,
    EngineEventKind,
    GenStats,
    Handle,
    ParkResult,
    RestoreResult,
)
from phaseweaver.clock import Clock
from phaseweaver.errors import EngineOutOfMemory
from phaseweaver.journal import EventJournal
from phaseweaver.types import Disposition


@dataclass
class _Allocation:
    tokens: int
    blocks: int
    offloaded: bool = False


class MockBackend:
    def __init__(
        self,
        caps: BackendCaps,
        clock: Clock,
        journal: EventJournal | None = None,
        *,
        run_id: str = "r1",
    ) -> None:
        self._caps = caps
        self._clock = clock
        self._journal = journal
        self._run_id = run_id
        self._allocs: dict[str, _Allocation] = {}
        self._events: list[EngineEvent] = []
        self._used_blocks = 0

    # --- protocol ---------------------------------------------------------
    def caps(self) -> BackendCaps:
        return self._caps

    def available_bytes(self) -> int:
        return self._caps.capacity_bytes - self._used_blocks * self._block_bytes()

    def engine_events(self) -> list[EngineEvent]:
        return list(self._events)

    async def submit(self, unit_id: str, context_id: str, tokens: int) -> Handle:
        handle = Handle(unit_id=unit_id, context_id=context_id, resident_tokens=tokens)
        self._allocate(context_id, tokens)
        return handle

    async def complete(self, handle: Handle, out_tokens: int) -> GenStats:
        start = self._clock.now()
        recompute = 0
        if handle.dropped:
            recompute = handle.resident_tokens
            self._allocate(handle.context_id, handle.resident_tokens)
            handle.dropped = False
        prefill = (handle.resident_tokens + recompute) / self._caps.prefill_tok_per_s
        decode = out_tokens / self._caps.decode_tok_per_s
        await self._clock.wait(prefill + decode)
        end = self._clock.now()
        return GenStats(
            prompt_tokens=handle.resident_tokens + recompute,
            completion_tokens=out_tokens,
            start_ts=start,
            end_ts=end,
            ttft_ms=prefill * 1000,
            recompute_tokens=recompute,
        )

    async def abort(self, handle: Handle, reason: str) -> None:
        self._free(handle.context_id)
        handle.dropped = True

    async def park(self, handle: Handle, disposition: Disposition) -> ParkResult:
        start = self._clock.now()
        alloc = self._allocs.get(handle.context_id)
        if alloc is None:
            return ParkResult(disposition, None, 0, 0.0)
        applied = disposition
        if applied is Disposition.OFFLOAD and not self._caps.supports_offload:
            applied = Disposition.DROP
        if applied is Disposition.RETAIN:
            return ParkResult(Disposition.RETAIN, None, 0, 0.0)
        if applied is Disposition.RELEASE:
            self._free(handle.context_id)
            return ParkResult(Disposition.RELEASE, None, 0, 0.0)
        moved = alloc.tokens * self._caps.bytes_per_token
        self._free(handle.context_id)
        handle.dropped = True
        self._emit(
            EngineEventKind.REQUEST_PREEMPTED,
            handle.context_id,
            0,
            {"disposition": applied.value},
        )
        if applied is Disposition.OFFLOAD:
            await self._clock.wait(moved / self._caps.mem_bw_bps)
        return ParkResult(
            applied=applied,
            degraded_from=disposition if applied is not disposition else None,
            moved_bytes=moved if applied is Disposition.OFFLOAD else 0,
            latency_ms=(self._clock.now() - start) * 1000,
        )

    async def restore(self, handle: Handle) -> RestoreResult:
        start = self._clock.now()
        recompute = 0
        alloc = self._allocs.get(handle.context_id)
        if alloc is None or handle.dropped:
            recompute = handle.resident_tokens
            self._allocate(handle.context_id, recompute)
            await self._clock.wait(recompute / self._caps.prefill_tok_per_s)
        handle.dropped = False
        return RestoreResult(
            recompute_tokens=recompute,
            loaded_bytes=0,
            latency_ms=(self._clock.now() - start) * 1000,
        )

    # --- internals --------------------------------------------------------
    def _block_bytes(self) -> int:
        return self._caps.block_tokens * self._caps.bytes_per_token

    def _allocate(self, context_id: str, tokens: int) -> None:
        blocks = (tokens + self._caps.block_tokens - 1) // self._caps.block_tokens
        if self._used_blocks + blocks > self._caps.blocks_total:
            raise EngineOutOfMemory(
                f"blocks exhausted: used={self._used_blocks} want={blocks} "
                f"cap={self._caps.blocks_total}"
            )
        self._allocs[context_id] = _Allocation(tokens=tokens, blocks=blocks)
        self._used_blocks += blocks
        self._emit(EngineEventKind.BLOCK_STORED, context_id, blocks)

    def _free(self, context_id: str) -> None:
        alloc = self._allocs.pop(context_id, None)
        if alloc is not None:
            self._used_blocks -= alloc.blocks
            self._emit(EngineEventKind.BLOCK_REMOVED, context_id, alloc.blocks)

    def _emit(
        self,
        kind: EngineEventKind,
        context_id: str | None,
        blocks: int,
        payload: dict[str, object] | None = None,
    ) -> None:
        self._events.append(
            EngineEvent(kind=kind, context_id=context_id, blocks=blocks, payload=payload)
        )
