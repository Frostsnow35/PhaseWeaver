"""B2: global SLO scheduler (ADR-0005).

Decisions use only the read-only snapshot plus its own history (park cooldown,
thrash window, DRR deficit).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from phaseweaver.policy import ClusterSnapshot
from phaseweaver.types import Disposition

_LADDER = (Disposition.RETAIN, Disposition.OFFLOAD, Disposition.DROP)


@dataclass
class SchedulerConfig:
    retain_max_s: float = 2.0
    offload_max_s: float = 20.0
    pressure_threshold: float = 0.8
    min_residency_s: float = 5.0
    park_cooldown_s: float = 10.0
    hysteresis_band: float = 0.15
    thrash_window_s: float = 60.0
    thrash_threshold: int = 2
    quantum_slots: int = 1
    allow_offload: bool = True


@dataclass
class _UnitState:
    last_park_ts: float = -1e9
    last_restore_ts: float = -1e9
    park_count_window: int = 0
    deficit: int = 0
    resident_since: float = 0.0


@dataclass
class SystemScheduler:
    config: SchedulerConfig = field(default_factory=SchedulerConfig)
    _units: dict[str, _UnitState] = field(default_factory=dict)

    def _state(self, unit_id: str) -> _UnitState:
        return self._units.setdefault(unit_id, _UnitState())

    # --- protocol ---------------------------------------------------------
    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        return snap.pressure < 0.98

    def allow_restore(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        """Only capacity can refuse a restore; the park cooldown must not.

        Conflating the two starved waiters in E1 (3 tasks failed). Cooldown is a
        *parking* brake, not a *restoring* brake (F3).
        """
        return snap.pressure < 0.98

    def disposition_for(
        self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float
    ) -> Disposition:
        cfg = self.config
        st = self._state(unit_id)

        # 1) thrash guard: repeated park/restore -> force retention
        if st.park_count_window >= cfg.thrash_threshold:
            return Disposition.RETAIN
        # 2) cooldown: parked recently, do not park again
        if snap.now - st.last_park_ts < cfg.park_cooldown_s:
            return Disposition.RETAIN
        # 3) minimum residency
        if (
            snap.now - st.resident_since < cfg.min_residency_s
            and snap.pressure < cfg.pressure_threshold
        ):
            return Disposition.RETAIN

        slack = snap.slack_by_unit.get(unit_id, 1e9)

        # 4) SLO already unreachable: do not preempt others on its behalf
        if slack < 0:
            return Disposition.DROP
        # 5) very urgent: keep
        if slack < expected_tool_s:
            return Disposition.RETAIN
        # 6) relaxed: shift one level down
        base = self._base(expected_tool_s)
        if slack > 2.0 * max(expected_tool_s, 1e-6):
            base = self._shift(base, 1)
        # 7) capacity pressure: shift one level down
        if snap.pressure > cfg.pressure_threshold:
            base = self._shift(base, 1)
        return self._degrade_if_needed(base)

    # --- bookkeeping ------------------------------------------------------
    def record_park(self, unit_id: str, now: float) -> None:
        st = self._state(unit_id)
        st.last_park_ts = now
        st.park_count_window += 1

    def record_restore(self, unit_id: str, now: float) -> None:
        st = self._state(unit_id)
        st.last_restore_ts = now

    def note_activated(self, unit_id: str, now: float) -> None:
        st = self._state(unit_id)
        st.resident_since = now

    def tick(self, now: float) -> None:
        """Advance one slot: slide the thrash window and accrue DRR deficit."""
        for st in self._units.values():
            if now - st.last_restore_ts > self.config.thrash_window_s:
                st.park_count_window = 0
            st.deficit += self.config.quantum_slots

    # --- internals --------------------------------------------------------
    def _base(self, expected_tool_s: float) -> Disposition:
        if expected_tool_s < self.config.retain_max_s:
            return Disposition.RETAIN
        if self.config.allow_offload and expected_tool_s < self.config.offload_max_s:
            return Disposition.OFFLOAD
        return Disposition.DROP

    @staticmethod
    def _shift(d: Disposition, steps: int) -> Disposition:
        idx = min(max(_LADDER.index(d) + steps, 0), len(_LADDER) - 1)
        return _LADDER[idx]

    def _degrade_if_needed(self, d: Disposition) -> Disposition:
        if d is Disposition.OFFLOAD and not self.config.allow_offload:
            return Disposition.DROP
        return d
