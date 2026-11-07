"""Autopilot: a pressure-adaptive policy that closes the loop with the bound.

The insight from the optimality work is that the device is limited by *how many
streams are generating at once*, and that limit depends on how much capacity is
being held by agents that are currently in a tool phase. So the policy should not
use a fixed aggression level: it should watch queue pressure and widen or narrow
the yield thresholds with hysteresis, and it should back off when a context just
thrashed.

This subclasses `SystemScheduler` and only adapts its thresholds, so every safety
property (cooldown, thrash promotion, SLO give-up) is inherited unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from phaseweaver.policy import ClusterSnapshot
from phaseweaver.scheduler import SchedulerConfig, SystemScheduler
from phaseweaver.types import Disposition


@dataclass
class AutopilotConfig:
    #: EWMA smoothing for the observed pressure and queue
    alpha: float = 0.35
    #: above this, yield more aggressively
    hot_pressure: float = 0.80
    #: below this and with no queue, keep contexts warm
    cool_pressure: float = 0.55
    queue_hot: float = 0.5
    retain_when_cool_s: float = 3.0
    retain_when_hot_s: float = 0.5


class AutopilotPolicy(SystemScheduler):
    def __init__(
        self,
        config: SchedulerConfig | None = None,
        acfg: AutopilotConfig | None = None,
    ) -> None:
        super().__init__(config or SchedulerConfig())
        self.acfg = acfg or AutopilotConfig()
        self._pressure = 0.0
        self._queue = 0.0

    def _observe(self, snap: ClusterSnapshot) -> None:
        a = self.acfg.alpha
        self._pressure = (1 - a) * self._pressure + a * snap.pressure
        self._queue = (1 - a) * self._queue + a * float(snap.queued)

    def _adapt(self) -> None:
        hot = self._pressure >= self.acfg.hot_pressure or self._queue >= self.acfg.queue_hot
        cool = self._pressure <= self.acfg.cool_pressure and self._queue < 0.5
        if hot:
            # Under pressure: give memory back sooner, so more streams can run.
            self.config.retain_max_s = self.acfg.retain_when_hot_s
            self.config.pressure_threshold = 0.6
        elif cool:
            # Nothing waiting: keeping the KV warm is free and saves a re-prefill.
            self.config.retain_max_s = self.acfg.retain_when_cool_s
            self.config.pressure_threshold = 0.9

    def disposition_for(
        self, unit_id: str, snap: ClusterSnapshot, expected_tool_s: float
    ) -> Disposition:
        self._observe(snap)
        self._adapt()
        return super().disposition_for(unit_id, snap, expected_tool_s)

    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        self._observe(snap)
        return super().admit(unit_id, snap)
