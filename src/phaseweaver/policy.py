"""Scheduling policies. They only read snapshots; state changes go through Runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from phaseweaver.types import Disposition


@dataclass(frozen=True)
class ClusterSnapshot:
    now: float
    capacity_bytes: int
    resident_bytes: int
    queued: int
    reasoning: int
    slack_by_unit: dict[str, float]
    recent_thrash: dict[str, int]
    # --- cost-model inputs (K3) ---
    resident_bytes_by_unit: dict[str, int] = field(default_factory=dict)
    tokens_by_unit: dict[str, int] = field(default_factory=dict)
    prefill_tok_per_s: int = 3200
    mem_bw_bps: int = 12 * 1024**3

    @property
    def pressure(self) -> float:
        if self.capacity_bytes == 0:
            return 1.0
        return self.resident_bytes / self.capacity_bytes


class Scheduler(Protocol):
    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool: ...
    def disposition_for(
        self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float
    ) -> Disposition: ...
    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool: ...


class PinAlwaysPolicy:
    """B0: always keep, never yield, never preempt."""

    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return True

    def disposition_for(
        self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float
    ) -> Disposition:
        return Disposition.RETAIN

    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return True


class TtlEvictPolicy:
    """B1: drop once the tool runs longer than a fixed TTL (Continuum-style)."""

    def __init__(self, ttl_s: float = 5.0) -> None:
        self.ttl_s = ttl_s

    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return True

    def disposition_for(
        self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float
    ) -> Disposition:
        return Disposition.RETAIN if expected_tool_s < self.ttl_s else Disposition.DROP

    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return True


class YieldOnToolPolicy:
    """B2a: per-unit duration buckets (mechanism bring-up and ablation baseline)."""

    def __init__(
        self,
        retain_max_s: float = 2.0,
        offload_max_s: float = 20.0,
        *,
        offload_available: bool = True,
        pressure_threshold: float = 0.8,
    ) -> None:
        self.retain_max_s = retain_max_s
        self.offload_max_s = offload_max_s
        self.offload_available = offload_available
        self.pressure_threshold = pressure_threshold

    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return True

    def _base(self, expected_tool_s: float) -> Disposition:
        if expected_tool_s < self.retain_max_s:
            return Disposition.RETAIN
        if self.offload_available and expected_tool_s < self.offload_max_s:
            return Disposition.OFFLOAD
        return Disposition.DROP

    def disposition_for(
        self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float
    ) -> Disposition:
        base = self._base(expected_tool_s)
        if snap.pressure <= self.pressure_threshold:
            return base
        ladder = [Disposition.RETAIN, Disposition.OFFLOAD, Disposition.DROP]
        degraded = ladder[min(ladder.index(base) + 1, len(ladder) - 1)]
        if degraded is Disposition.OFFLOAD and not self.offload_available:
            return Disposition.DROP
        return degraded

    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return snap.pressure < 0.98
