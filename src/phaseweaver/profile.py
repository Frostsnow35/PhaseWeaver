"""Profiles: a scenario is a bundle of defaults, not just documentation.

A profile answers "how should the runtime behave for this kind of workload" so the
quickstart does not require understanding leases, dispositions and scheduler
knobs. Everything a profile sets can still be overridden explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    #: scheduler preset: "auto" (pressure-adaptive), "B2" (global SLO), "B0"
    policy: str = "auto"
    #: default KV inheritance for forks: "none" | "handoff" | "relay"
    inherit_mode: str = "none"
    #: consider this fraction of capacity unusable (growth, fragmentation)
    headroom_frac: float = 0.05
    #: default SLO per task, seconds
    slo_s: float = 240.0
    #: extra scheduler overrides applied on top of the preset
    scheduler: dict[str, float] = field(default_factory=dict)


CODING = Profile(
    name="coding",
    description="编码 Agent:短工具、长会话,上下文值得保留",
    policy="auto",
    inherit_mode="none",
    slo_s=240.0,
)

RESEARCH = Profile(
    name="research",
    description="检索/浏览 Agent:长工具,GPU 空闲期明显",
    policy="B2",
    inherit_mode="none",
    slo_s=240.0,
)

ORCHESTRATOR = Profile(
    name="orchestrator",
    description="多 Agent 编排:扇出/汇合,子任务需要父上下文",
    policy="B2",
    inherit_mode="handoff",
    slo_s=240.0,
)

DELEGATION = Profile(
    name="delegation",
    description="链式委托:父→子→孙,上下文逐跳继承",
    policy="B2",
    inherit_mode="handoff",
    slo_s=240.0,
)

PROFILES: dict[str, Profile] = {
    p.name: p for p in (CODING, RESEARCH, ORCHESTRATOR, DELEGATION)
}


def resolve(profile: str | Profile) -> Profile:
    if isinstance(profile, Profile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise KeyError(
            f"unknown profile {profile!r}; available: {sorted(PROFILES)}"
        ) from exc
