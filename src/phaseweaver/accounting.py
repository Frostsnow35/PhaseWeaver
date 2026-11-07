"""Pure accounting helpers for the runtime.

Keeping these out of `runtime.py` matters for two reasons: they are trivially
testable without a backend, and the orchestration code stays small enough to hold
in one's head (our own rule: <= 400 lines per file).
"""

from __future__ import annotations

from collections.abc import MutableMapping

from phaseweaver.backends.base import BackendCaps
from phaseweaver.leases import LeaseBroker
from phaseweaver.types import ContextState, Disposition, LeaseKind

#: how many park events inside the window turn into a thrash signal
THRASH_WINDOW_DEFAULT_S = 60.0


def lease_bytes(caps: BackendCaps, tokens: int) -> int:
    """Round up to engine block granularity so budget and allocator agree."""
    blocks = (tokens + caps.block_tokens - 1) // caps.block_tokens
    return blocks * caps.block_tokens * caps.bytes_per_token


def max_tokens_for(caps: BackendCaps, broker: LeaseBroker, scope: str | None) -> int | None:
    """Largest context the unit's own ISOLATE quota can hold, in tokens."""
    limit = broker.scope_limit(scope)
    if limit is None:
        return None
    block_bytes = caps.block_tokens * caps.bytes_per_token
    blocks = limit // block_bytes
    return max(caps.block_tokens, blocks * caps.block_tokens)


_STATE_FOR: dict[Disposition, ContextState] = {
    Disposition.RETAIN: ContextState.RETAINED,
    Disposition.DROP: ContextState.DROPPED,
    Disposition.OFFLOAD: ContextState.OFFLOADED,
    Disposition.RELEASE: ContextState.RELEASED,
}


def state_for(disposition: Disposition) -> ContextState:
    return _STATE_FOR[disposition]


def record_park(
    history: MutableMapping[str, list[float]],
    unit_id: str,
    now: float,
    window_s: float = THRASH_WINDOW_DEFAULT_S,
) -> int:
    """Append a park event, drop entries outside the window, return the count."""
    recent = [ts for ts in history.get(unit_id, []) if now - ts <= window_s]
    recent.append(now)
    history[unit_id] = recent
    return len(recent)


def parked_lease_kind() -> LeaseKind:
    return LeaseKind.KV_BYTES
