"""Third-party extension discovery via setuptools entry points.

Groups:

    phaseweaver.backends   name -> InferenceBackend subclass/factory
    phaseweaver.policies   name -> Scheduler class/factory
    phaseweaver.tools      name -> ToolExecutor instance

Loading is deliberately lazy and explicit: importing `phaseweaver` never imports
plugin code, and a broken plugin produces a clear error instead of a mystery
ImportError during a run.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from phaseweaver.errors import PluginError

BACKENDS = "phaseweaver.backends"
POLICIES = "phaseweaver.policies"
TOOLS = "phaseweaver.tools"

GROUPS = (BACKENDS, POLICIES, TOOLS)


def _entry_points_for(group: str) -> list[Any]:
    """Cross-version entry point lookup (3.10 select API vs 3.9 dict)."""
    try:
        selected = importlib.metadata.entry_points(group=group)
    except TypeError:  # pragma: no cover - Python 3.9
        legacy = importlib.metadata.entry_points()
        selected = legacy.get(group, ())  # type: ignore[arg-type]
    return list(selected)


def load_group(group: str) -> dict[str, Any]:
    """Load every entry point in `group`; raise PluginError with context on failure."""
    loaded: dict[str, Any] = {}
    for entry_point in _entry_points_for(group):
        try:
            loaded[entry_point.name] = entry_point.load()
        except Exception as exc:  # noqa: BLE001 - report which plugin is broken
            raise PluginError(
                f"failed to load plugin {entry_point.name!r} from group {group!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return loaded


def resolve_backend(name: str) -> Any:
    plugins = load_group(BACKENDS)
    if name not in plugins:
        known = sorted(plugins)
        raise PluginError(
            f"unknown backend {name!r}; installed backends: {known or 'none'}"
        )
    return plugins[name]


def resolve_policy(name: str) -> Any:
    plugins = load_group(POLICIES)
    if name not in plugins:
        known = sorted(plugins)
        raise PluginError(
            f"unknown policy {name!r}; installed policies: {known or 'none'}"
        )
    return plugins[name]
