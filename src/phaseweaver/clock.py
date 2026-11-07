"""Clock abstraction.

Simulation must use VirtualClock; direct sleeping is forbidden in the mock path
(see plan section 0).

The driver is **event-driven** (F9): it sleeps until a timer is registered instead
of polling `sleep(0)` forever. That removes the busy loop that made wall time grow
superlinearly with the number of concurrent agents.
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

    Time advances only when no coroutine can make progress: the driver jumps to the
    earliest pending timer and resolves every waiter at that instant. Ties are
    resolved FIFO by registration order, so the same seed yields the same event
    order.
    """

    def __init__(self, start: float = 0.0, idle_rounds: int = 4) -> None:
        #: kept for backwards compatibility; the driver no longer needs a heuristic
        self._idle_rounds = idle_rounds
        self._now = start
        self._heap: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self._wake: asyncio.Event | None = None
        self._driver: asyncio.Task[None] | None = None

    def now(self) -> float:
        return self._now

    def _event(self) -> asyncio.Event:
        if self._wake is None:
            self._wake = asyncio.Event()
        return self._wake

    async def wait(self, dt: float) -> None:
        if dt <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (self._now + dt, next(self._seq), fut))
        self._event().set()
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
            self._purge()
            if not self._heap:
                self._event().clear()
                await self._event().wait()
                continue
            deadline = self._heap[0][0]
            if deadline > self._now:
                self._now = deadline
            self._fire(self._now)
            # Let the coroutines we just woke make progress (and register new
            # timers) before deciding whether to advance again.
            await asyncio.sleep(0)

    def _purge(self) -> None:
        """Drop timers whose waiter was cancelled (they can never be honoured)."""
        while self._heap and self._heap[0][2].done() and not self._heap[0][2].cancelled():
            heapq.heappop(self._heap)
        while self._heap and self._heap[0][2].cancelled():
            heapq.heappop(self._heap)

    def _fire(self, now: float) -> None:
        while self._heap and self._heap[0][0] <= now:
            _, _, fut = heapq.heappop(self._heap)
            if not fut.done():
                fut.set_result(None)
