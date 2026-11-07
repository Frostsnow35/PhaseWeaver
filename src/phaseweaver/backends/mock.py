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
        self._active = 0

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
        # F2: all concurrent decodes share one device; per-stream rate degrades
        # with the number of active streams (batching efficiency alpha).
        self._active += 1
        try:
            if handle.dropped:
                recompute = handle.resident_tokens
                self._allocate(handle.context_id, handle.resident_tokens)
                handle.dropped = False
            prompt = handle.resident_tokens + recompute
            await self._run_tokens(prompt, self._caps.prefill_tok_per_s)
            await self._run_tokens(out_tokens, self._caps.decode_tok_per_s)
            self._grow(handle.context_id, handle.resident_tokens + out_tokens)
            handle.resident_tokens += out_tokens
        finally:
            self._active -= 1
        end = self._clock.now()
        return GenStats(
            prompt_tokens=prompt,
            completion_tokens=out_tokens,
            start_ts=start,
            end_ts=end,
            ttft_ms=(end - start) * 1000,
            recompute_tokens=recompute,
        )

    async def _run_tokens(self, tokens: float, base_rate: float) -> None:
        """Advance up to `tokens` tokens of work at the current shared rate."""
        remaining = float(tokens)
        dt = max(1e-3, self._caps.time_slice_s)
        while remaining > 0:
            streams = max(1, self._active)
            rate = base_rate / (1.0 + self._caps.contention_alpha * (streams - 1))
            remaining -= max(1.0, rate * dt)
            await self._clock.wait(dt)

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

    async def handoff(
        self, handle: Handle, new_unit_id: str, new_context_id: str
    ) -> Handle:
        """Move an allocation's ownership to another unit (KV handoff, zero copy)."""
        alloc = self._allocs.pop(handle.context_id, None)
        new_handle = Handle(
            unit_id=new_unit_id,
            context_id=new_context_id,
            resident_tokens=handle.resident_tokens,
            dropped=handle.dropped,
        )
        if alloc is not None:
            self._allocs[new_context_id] = alloc
        handle.dropped = True
        return new_handle

    async def truncate(self, handle: Handle, tokens: int) -> int:
        """Shrink the allocation in place; the dropped blocks return to the pool."""
        target = max(1, min(tokens, handle.resident_tokens))
        blocks = (target + self._caps.block_tokens - 1) // self._caps.block_tokens
        alloc = self._allocs.get(handle.context_id)
        if alloc is not None:
            delta = alloc.blocks - blocks
            if delta > 0:
                alloc.blocks = blocks
                alloc.tokens = target
                self._used_blocks -= delta
                self._emit(EngineEventKind.BLOCK_REMOVED, handle.context_id, delta)
        handle.resident_tokens = target
        return target

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

    def _grow(self, context_id: str, tokens: int) -> None:
        """Reserve the extra blocks a longer context needs (delta only)."""
        alloc = self._allocs.get(context_id)
        if alloc is None:
            return
        own_tokens = tokens
        blocks = (own_tokens + self._caps.block_tokens - 1) // self._caps.block_tokens
        delta = blocks - alloc.blocks
        alloc.tokens = tokens
        if delta <= 0:
            return
        if self._used_blocks + delta > self._caps.blocks_total:
            raise EngineOutOfMemory(
                f"blocks exhausted on grow: used={self._used_blocks} want={delta} "
                f"cap={self._caps.blocks_total}"
            )
        alloc.blocks = blocks
        self._used_blocks += delta
        self._emit(EngineEventKind.BLOCK_STORED, context_id, delta)

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
