"""Exception hierarchy."""

from __future__ import annotations


class PhaseWeaverError(Exception):
    """Base class for all PhaseWeaver errors."""


class IllegalTransition(PhaseWeaverError):
    def __init__(self, kind: str, src: str, dst: str) -> None:
        super().__init__(f"illegal {kind} transition: {src} -> {dst}")
        self.kind = kind
        self.src = src
        self.dst = dst


class AdmissionRejected(PhaseWeaverError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"admission rejected: {reason}")
        self.reason = reason


class BudgetExceeded(PhaseWeaverError):
    pass


class JoinBeforeTermination(PhaseWeaverError):
    pass


class ToolTimeout(PhaseWeaverError):
    pass


class ReapIncomplete(PhaseWeaverError):
    pass


class EngineOutOfMemory(PhaseWeaverError):
    """Runtime accounting bug: lease allowed but the engine cannot fit the blocks."""


class RuntimeClosed(PhaseWeaverError):
    """The runtime is not running: start it (or use the PhaseWeaver facade)."""


class PluginError(PhaseWeaverError):
    """A third-party entry point is missing or failed to import."""


class ConfigError(PhaseWeaverError):
    pass
