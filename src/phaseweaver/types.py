"""Core value types. This module must not import other internal modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

RunId = str
UnitId = str
ContextId = str
LeaseId = str


class Phase(StrEnum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    REASONING = "REASONING"
    ACTING = "ACTING"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def is_terminal(self) -> bool:
        return self in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED)


class ContextState(StrEnum):
    NONE = "NONE"
    ACTIVE = "ACTIVE"
    RETAINED = "RETAINED"
    OFFLOADED = "OFFLOADED"
    DROPPED = "DROPPED"
    #: ownership moved to another unit (KV handoff); no resident blocks, no lease
    HANDED_OFF = "HANDED_OFF"
    RELEASED = "RELEASED"


class UnitKind(StrEnum):
    AGENT = "agent"
    SUBAGENT = "subagent"
    TOOL_CALL = "tool_call"


class LeaseKind(StrEnum):
    KV_BYTES = "kv_bytes"
    OFFLOAD_BYTES = "offload_bytes"
    ENGINE_SLOT = "engine_slot"
    TOOL_SLOT = "tool_slot"


class Disposition(StrEnum):
    RETAIN = "retain"
    OFFLOAD = "offload"
    DROP = "drop"
    RELEASE = "release"


class ForkMode(StrEnum):
    SHARE = "share"
    ISOLATE = "isolate"


@dataclass
class InferenceContext:
    context_id: ContextId
    state: ContextState = ContextState.NONE
    resident_tokens: int = 0
    resident_bytes: int = 0
    offload_bytes: int = 0
    updated_seq: int | None = None
    #: KV handoff: the unit that currently owns this context's blocks, if any
    handed_off_to: str | None = None

    @property
    def occupies_gpu(self) -> bool:
        return self.state in (ContextState.ACTIVE, ContextState.RETAINED)

    @property
    def needs_recompute(self) -> bool:
        return self.state is ContextState.DROPPED


@dataclass
class UnitAggregate:
    """Accounting merged on join."""

    tasks_completed: int = 0
    recompute_tokens: int = 0
    park_count: int = 0
    child_ids: list[UnitId] = field(default_factory=list)
