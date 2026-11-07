"""Orchestrator example: parent delegates to a subagent, KV is handed off.

Run: python examples/orchestrator_agent.py

The orchestrator profile defaults to `handoff`, so the child starts from the
parent's KV instead of re-prefilling it, and join hands it back.
"""

from __future__ import annotations

import asyncio

from phaseweaver import PhaseWeaver, ToolSpec


async def child(unit) -> None:  # type: ignore[no-untyped-def]
    await unit.reason(prompt_tokens=6144, out_tokens=128)
    await unit.call_tool(
        ToolSpec("fetch", "python", expected_duration_s=6.0), {"code": "print('done')"}
    )


async def parent(unit) -> None:  # type: ignore[no-untyped-def]
    await unit.reason(prompt_tokens=4096, out_tokens=128)
    await unit.fork("researcher", child)
    await unit.reason(prompt_tokens=6400, out_tokens=128)


async def main() -> None:
    async with PhaseWeaver(profile="orchestrator", capacity_gib=2.0) as pw:
        await pw.run("investigate", parent)
        summary = pw.summary()
        handed = [
            e
            for e in pw.runtime.journal.by_type("context_state_changed")
            if e.payload.get("state") == "HANDED_OFF"
        ]
        print(f"kv_handoffs={len(handed)} p99={summary['p99_ms']:.0f}ms "
              f"goodput={summary['goodput']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
