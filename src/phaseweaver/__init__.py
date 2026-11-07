"""PhaseWeaver: a phase-aware agent execution runtime.

The public API has three tiers:

1. `PhaseWeaver` (this facade) - one object, scenario profile, five lines.
2. `Runtime` + `AgentHandle` - explicit control over backend, budget and policy.
3. Protocols - `InferenceBackend`, `Scheduler`, `ToolExecutor`, `Profile` - the
   extension points for new engines, policies, tools and scenarios.
"""

from __future__ import annotations

from phaseweaver.app import PhaseWeaver
from phaseweaver.app import run as run_task
from phaseweaver.backends.base import BackendCaps, Handle, InferenceBackend
from phaseweaver.leases import Budget, Lease, LeaseBroker
from phaseweaver.metrics import RunReport
from phaseweaver.policy import ClusterSnapshot, Scheduler
from phaseweaver.profile import PROFILES, Profile
from phaseweaver.runtime import AgentHandle, Runtime, RuntimeConfig, TaskOutcome
from phaseweaver.tools.base import ToolResult, ToolSpec, ToolStatus
from phaseweaver.types import (
    ContextState,
    Disposition,
    ForkMode,
    InferenceContext,
    LeaseKind,
    Phase,
    UnitKind,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # L1
    "PhaseWeaver",
    "run_task",
    "Profile",
    "PROFILES",
    # L2
    "Runtime",
    "RuntimeConfig",
    "AgentHandle",
    "TaskOutcome",
    "Budget",
    "Lease",
    "LeaseBroker",
    "RunReport",
    # types
    "Phase",
    "ContextState",
    "Disposition",
    "ForkMode",
    "InferenceContext",
    "LeaseKind",
    "UnitKind",
    # L3
    "InferenceBackend",
    "BackendCaps",
    "Handle",
    "Scheduler",
    "ClusterSnapshot",
    "ToolSpec",
    "ToolResult",
    "ToolStatus",
]
