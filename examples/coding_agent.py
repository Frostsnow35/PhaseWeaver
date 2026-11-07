"""Minimal coding-agent example: five-line entry point, simulated backend.

Run: python examples/coding_agent.py
"""

from __future__ import annotations

import asyncio

from phaseweaver import PhaseWeaver, ToolSpec


async def agent(unit) -> None:  # type: ignore[no-untyped-def]
    prompt = 2048
    for _ in range(3):
        await unit.reason(prompt_tokens=prompt, out_tokens=128)
        await unit.call_tool(
            ToolSpec("pytest", "python", expected_duration_s=4.0),
            {"code": "print('tests pass')"},
        )
        prompt += 128


async def main() -> None:
    async with PhaseWeaver(profile="coding", capacity_gib=2.0) as pw:
        outcome = await pw.run("fix-bug", agent)
        print(f"outcome={outcome.phase} latency={outcome.latency_ms / 1000:.1f}s")
        print(pw.summary())


if __name__ == "__main__":
    asyncio.run(main())
