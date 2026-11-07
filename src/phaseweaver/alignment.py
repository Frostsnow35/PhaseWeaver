"""R6: phase alignment by stream-slot admission.

The hypothesis: with *heterogeneous* tool durations, admitting a unit only when a
reasoning slot is free desynchronises tool windows, keeping the device busier than
capacity-driven admission alone.

The duty-cycle bound says this cannot help when every agent has the same r/t ratio:
`c̄ <= k·r/(r+t)`, and uniform agents already realise it. The interesting case is a
mix of short-tool and long-tool agents, where the *instantaneous* overlap can dip
below the bound. This policy exists to test that case honestly - it is not enabled
by default, and the experiment records a negative result if it does not pay off.
"""

from __future__ import annotations

from phaseweaver.policy import ClusterSnapshot
from phaseweaver.scheduler import SchedulerConfig, SystemScheduler


class AlignedPolicy(SystemScheduler):
    """B2 + admission gate: at most `target_streams` units may reason at once."""

    def __init__(
        self,
        target_streams: int = 2,
        config: SchedulerConfig | None = None,
    ) -> None:
        super().__init__(config or SchedulerConfig())
        self.target_streams = target_streams

    def admit(self, unit_id: str, snap: ClusterSnapshot) -> bool:
        if snap.pressure >= 0.98:
            return False
        return snap.reasoning < self.target_streams
