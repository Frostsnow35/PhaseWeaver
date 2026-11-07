"""KV prefix sharing across an agent tree ("KV fork").

The idea: a subagent usually starts from (a superset of) its parent's context.
Copying those blocks wastes memory; dropping them wastes prefill. Instead the
child *references* the parent's resident prefix and pays only for the delta.

Accounting rules:
- a prefix group is charged **once** against the budget, regardless of how many
  units reference it (that is why the broker deduplicates by `prefix_key`);
- every referencing context still owns a lease for I2 (it does occupy the GPU),
  but the lease records `charged_bytes`, which is 0 for fully shared residency.

This module is deliberately small and dependency-free so the semantics can be
property-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrefixGroup:
    key: str
    tokens: int
    bytes_: int
    refs: int = 1
    owners: set[str] = field(default_factory=set)


class PrefixRegistry:
    """Refcounted, per-key registry of shared KV prefixes."""

    def __init__(self) -> None:
        self._groups: dict[str, PrefixGroup] = {}

    def acquire(self, key: str, tokens: int, bytes_: int, owner: str) -> PrefixGroup:
        group = self._groups.get(key)
        if group is None:
            group = PrefixGroup(key=key, tokens=tokens, bytes_=bytes_)
            self._groups[key] = group
        else:
            group.refs += 1
        group.owners.add(owner)
        return group

    def release(self, key: str, owner: str) -> bool:
        """Returns True when the group became empty and its bytes were freed."""
        group = self._groups.get(key)
        if group is None:
            return False
        group.owners.discard(owner)
        group.refs = max(0, group.refs - 1)
        if group.refs == 0:
            del self._groups[key]
            return True
        return False

    def get(self, key: str) -> PrefixGroup | None:
        return self._groups.get(key)

    def unique_bytes(self) -> int:
        return sum(g.bytes_ for g in self._groups.values())

    def groups(self) -> list[PrefixGroup]:
        return list(self._groups.values())
