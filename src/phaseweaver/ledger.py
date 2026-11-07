"""Per-unit resource state, in one place.

Before this existed, `Runtime` kept three parallel dicts keyed by unit id
(`_handles`, `_leases`, `_park_times`). Every bug of the K15 family came from those
dicts drifting apart. Grouping them makes the invariant obvious:

    a unit with a handle but no lease (or the reverse) is a bug, not a state.

The container is intentionally dict-like: call sites stay readable, and a future
`handoff.py` can take the whole ledger as its narrow interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phaseweaver.leases import Lease
from phaseweaver.types import UnitId


@dataclass
class UnitLedger:
    #: engine handles owned by a unit (moved during KV handoff)
    handles: dict[UnitId, Any] = field(default_factory=dict)
    #: KV leases owned by a unit (transferred, not re-acquired, on handoff)
    leases: dict[UnitId, Lease] = field(default_factory=dict)
    #: recent park timestamps, used for thrash detection
    park_times: dict[UnitId, list[float]] = field(default_factory=dict)

    def clear(self) -> None:
        self.handles.clear()
        self.leases.clear()
        self.park_times.clear()

    def owner_of_handle(self, handle: Any) -> UnitId | None:
        for unit_id, candidate in self.handles.items():
            if candidate is handle:
                return unit_id
        return None
