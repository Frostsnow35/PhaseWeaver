"""Stable capability-report schema for inference-engine adapters.

The problem this file solves: two engine adapters both produced a flat
``dict[str, object]`` with a field named ``kv_pool_bytes``. In one it was a
deployment declaration, in the other an estimate derived from an assumed
``bytes_per_token``. Same key, different evidence strength. Downstream reports
could not tell them apart.

Every field therefore carries a :class:`CapabilityOrigin`, and ``as_dict()``
emits ``<name>_origin`` alongside each value. ``BackendCaps`` values are always
``DECLARED`` (they come from config), never ``PROBED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from phaseweaver.backends.base import BackendCaps


class CapabilityOrigin(StrEnum):
    PROBED = "probed"        # observed over the wire from the running engine
    DECLARED = "declared"    # stated by deployment config / CLI flags
    ESTIMATED = "estimated"  # derived from a model assumption, not measured
    UNKNOWN = "unknown"      # engine did not expose this; never invent a value


@dataclass(frozen=True)
class CapabilityField:
    name: str
    value: object
    origin: CapabilityOrigin
    detail: str = ""


@dataclass(frozen=True)
class CapabilityReport:
    engine: str
    #: bump when endpoint shapes change; reports from different versions are not comparable
    contract_version: str
    evidence: str
    probed: bool
    fields: tuple[CapabilityField, ...] = ()
    #: the engine release these endpoint shapes were checked against
    engine_version_checked: str = "unverified"
    detail: str = ""

    def field(self, name: str) -> CapabilityField | None:
        for item in self.fields:
            if item.name == name:
                return item
        return None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "engine": self.engine,
            "contract_version": self.contract_version,
            "engine_version_checked": self.engine_version_checked,
            "evidence": self.evidence,
            "probed": self.probed,
            "detail": self.detail,
        }
        for item in self.fields:
            out[item.name] = item.value
            out[f"{item.name}_origin"] = item.origin.value
            if item.detail:
                out[f"{item.name}_detail"] = item.detail
        return out


def caps_snapshot(
    caps: BackendCaps,
    *,
    engine: str,
    contract_version: str = "core-caps/1",
    evidence: str = "[L1] config-declared, not measured",
) -> CapabilityReport:
    """Wrap ``BackendCaps`` as declared fields so reports never imply measurement."""
    declared = CapabilityOrigin.DECLARED
    fields = (
        CapabilityField("capacity_bytes", caps.capacity_bytes, declared),
        CapabilityField("block_tokens", caps.block_tokens, declared),
        CapabilityField("bytes_per_token", caps.bytes_per_token, declared),
        CapabilityField("prefill_tok_per_s", caps.prefill_tok_per_s, declared),
        CapabilityField("decode_tok_per_s", caps.decode_tok_per_s, declared),
        CapabilityField("mem_bw_bps", caps.mem_bw_bps, declared),
        CapabilityField("contention_alpha", caps.contention_alpha, declared),
        CapabilityField("supports_offload", caps.supports_offload, declared),
        CapabilityField("supports_engine_sleep", caps.supports_engine_sleep, declared),
    )
    return CapabilityReport(
        engine=engine,
        contract_version=contract_version,
        evidence=evidence,
        probed=False,
        fields=fields,
        detail="BackendCaps is configuration, not an observation of the engine",
    )
