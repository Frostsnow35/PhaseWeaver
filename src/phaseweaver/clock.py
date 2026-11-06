"""Clock abstraction.

Simulation must use VirtualClock; direct sleeping is forbidden in the mock path
(see plan section 0).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    async def wait(self, dt: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return asyncio.get_running_loop().time()

    async def wait(self, dt: float) -> None:
        if dt > 0:
            await asyncio.sleep(dt)


class VirtualClock:
    """Discrete-event virtual clock.

    Time advances only when no coroutine can make progress, in which case the
    clock jumps to the earliest pending timer. Same seed => same event order.
    """

    def __init__(self, start: float = 0.0, idle_rounds: int = 4) -> None:
        self._now = start
        self._heap: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self._idle_rounds = idle_rounds
        self._wakeups = 0
        self._idle = 0
        self._last_wakeups = -1
        self._driver: asyncio.Task[None] | None = None

    def now(self) -> float:
        return self._now

    async def wait(self, dt: float) -> None:
        if dt <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (self._now + dt, next(self._seq), fut))
        self._wakeups += 1
        await fut

    def start_driver(self) -> None:
        if self._driver is None:
            self._driver = asyncio.create_task(self._drive(), name="virtual-clock")

    def stop_driver(self) -> None:
        if self._driver is not None:
            self._driver.cancel()
            self._driver = None

    async def _drive(self) -> None:
        while True:
            await asyncio.sleep(0)
            if not self._heap:
                self._last_wakeups = self._wakeups
                self._idle = 0
                continue
            if self._wakeups == self._last_wakeups:
                self._idle += 1
            else:
                self._idle = 0
                self._last_wakeups = self._wakeups
            if self._idle >= self._idle_rounds:
                self._jump()
                self._idle = 0

    def _jump(self) -> None:
        if not self._heap:
            return
        deadline = self._heap[0][0]
        self._now = deadline
        while self._heap and self._heap[0][0] <= deadline:
            _, _, fut = heapq.heappop(self._heap)
            if not fut.done():
                fut.set_result(None)
            self._wakeups += 1
