"""Plugin discovery tests (R5).

Entry points are faked so the test never depends on what is installed in the venv.
"""

from __future__ import annotations

from typing import Any

import pytest

from phaseweaver import PhaseWeaver
from phaseweaver.errors import PluginError
from phaseweaver.plugins import BACKENDS, POLICIES, load_group, resolve_backend, resolve_policy


class _FakeEntryPoint:
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self._value = value

    def load(self) -> Any:
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


def install(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, dict[str, Any]]) -> None:
    def fake(group: str) -> list[_FakeEntryPoint]:
        return [_FakeEntryPoint(name, value) for name, value in mapping.get(group, {}).items()]

    monkeypatch.setattr("phaseweaver.plugins._entry_points_for", fake)


def test_load_group_returns_named_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {BACKENDS: {"fake": object}})
    assert load_group(BACKENDS) == {"fake": object}


def test_broken_plugin_reports_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {BACKENDS: {"broken": RuntimeError("no native lib")}})
    with pytest.raises(PluginError, match="broken"):
        load_group(BACKENDS)


def test_unknown_plugin_lists_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, {BACKENDS: {"fake": object}})
    with pytest.raises(PluginError, match="installed backends"):
        resolve_backend("nope")
    with pytest.raises(PluginError, match="unknown policy"):
        resolve_policy("nope")


def test_facade_accepts_backend_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from phaseweaver.backends.mock import MockBackend
    from phaseweaver.clock import RealClock

    calls: list[str] = []

    def factory() -> MockBackend:
        calls.append("built")
        return MockBackend(caps=_small_caps(), clock=RealClock())

    install(monkeypatch, {BACKENDS: {"fake": factory}})
    pw = PhaseWeaver(backend="fake")
    assert calls == ["built"]
    assert isinstance(pw._backend, MockBackend)


def test_policy_plugin_is_used_by_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    from phaseweaver.policy import PinAlwaysPolicy

    class CustomPolicy(PinAlwaysPolicy):
        pass

    install(monkeypatch, {POLICIES: {"custom": CustomPolicy}})
    from phaseweaver.app import build_policy

    assert isinstance(build_policy("custom"), CustomPolicy)


def _small_caps():  # type: ignore[no-untyped-def]
    from phaseweaver.backends.base import BackendCaps

    return BackendCaps(
        capacity_bytes=1024 * 1024,
        block_tokens=16,
        bytes_per_token=1000,
        prefill_tok_per_s=1000,
        decode_tok_per_s=100,
        mem_bw_bps=10**7,
    )
