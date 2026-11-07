import asyncio

from phaseweaver import PhaseWeaver, Profile, __all__
from phaseweaver.profile import PROFILES


async def _agent(handle) -> None:  # type: ignore[no-untyped-def]
    await handle.reason(prompt_tokens=256, out_tokens=16)


def test_public_api_is_stable() -> None:
    # Guard rail: removing an export is a breaking change and must be deliberate.
    for name in (
        "PhaseWeaver",
        "Profile",
        "Runtime",
        "InferenceBackend",
        "Scheduler",
        "ToolSpec",
        "Phase",
        "Disposition",
    ):
        assert name in __all__


def test_facade_runs_a_task_and_reports() -> None:
    async def scenario() -> PhaseWeaver:
        async with PhaseWeaver(profile="coding", capacity_gib=2.0) as pw:
            outcome = await pw.run("demo", _agent)
            assert outcome.phase == "COMPLETED"
            return pw

    pw = asyncio.run(scenario())
    summary = pw.summary()
    assert summary["completed"] == 1.0
    assert summary["slo_rate"] == 1.0


def test_orchestrator_profile_applies_handoff_by_default() -> None:
    async def child(handle) -> None:  # type: ignore[no-untyped-def]
        await handle.reason(prompt_tokens=320, out_tokens=16)

    async def parent(handle) -> None:  # type: ignore[no-untyped-def]
        await handle.reason(prompt_tokens=256, out_tokens=16)
        await handle.fork("child", child)  # no explicit inherit: profile decides
        await handle.reason(prompt_tokens=320, out_tokens=16)

    async def scenario() -> PhaseWeaver:
        async with PhaseWeaver(profile="orchestrator", capacity_gib=2.0) as pw:
            await pw.run("delegation", parent)
            return pw

    pw = asyncio.run(scenario())
    handed = [
        e
        for e in pw.runtime.journal.by_type("context_state_changed")
        if e.payload.get("state") == "HANDED_OFF"
    ]
    assert handed, "orchestrator profile should default to KV handoff"


def test_unknown_profile_rejected() -> None:
    try:
        PhaseWeaver(profile="does-not-exist")
    except KeyError as exc:
        assert "available" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown profile must fail loudly")


def test_profiles_are_complete() -> None:
    for name, profile in PROFILES.items():
        assert profile.name == name
        assert profile.policy in {"auto", "B0", "B1", "B2", "B2a"}
        assert profile.inherit_mode in {"none", "handoff", "relay"}
    assert isinstance(Profile.__dataclass_fields__, dict)
