"""Append-only event journal: the single source of truth (no parallel counters)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phaseweaver.clock import Clock


@dataclass(frozen=True)
class Event:
    seq: int
    ts: float
    run_id: str
    event: str
    unit_id: str | None = None
    phase_from: str | None = None
    phase_to: str | None = None
    cause_seq: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "ts": self.ts,
                "run_id": self.run_id,
                "event": self.event,
                "unit_id": self.unit_id,
                "phase_from": self.phase_from,
                "phase_to": self.phase_to,
                "cause_seq": self.cause_seq,
                "payload": self.payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class EventJournal:
    def __init__(self, run_id: str, clock: Clock, path: Path | None = None) -> None:
        self.run_id = run_id
        self._clock = clock
        self._path = path
        self._events: list[Event] = []
        self._seq = 0
        self._fh: Any = path.open("w", encoding="utf-8", newline="\n") if path else None

    def append(
        self,
        event: str,
        *,
        unit_id: str | None = None,
        phase_from: str | None = None,
        phase_to: str | None = None,
        cause_seq: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        self._seq += 1
        ev = Event(
            seq=self._seq,
            ts=self._clock.now(),
            run_id=self.run_id,
            event=event,
            unit_id=unit_id,
            phase_from=phase_from,
            phase_to=phase_to,
            cause_seq=cause_seq,
            payload=payload or {},
        )
        self._events.append(ev)
        if self._fh is not None:
            self._fh.write(ev.to_json() + "\n")
            self._fh.flush()
        return ev

    def events(self) -> Iterator[Event]:
        return iter(self._events)

    def by_type(self, event: str) -> list[Event]:
        return [e for e in self._events if e.event == event]

    def digest(self) -> str:
        h = hashlib.sha256()
        for e in self._events:
            h.update(e.to_json().encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # --- replay from disk -------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> EventJournal:
        """Rebuild a journal from a JSONL file (for post-mortem analysis)."""
        events: list[Event] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            events.append(
                Event(
                    seq=int(raw["seq"]),
                    ts=float(raw["ts"]),
                    run_id=str(raw["run_id"]),
                    event=str(raw["event"]),
                    unit_id=raw.get("unit_id"),
                    phase_from=raw.get("phase_from"),
                    phase_to=raw.get("phase_to"),
                    cause_seq=raw.get("cause_seq"),
                    payload=raw.get("payload") or {},
                )
            )
        run_id = events[0].run_id if events else "loaded"
        journal = cls(run_id=run_id, clock=_FrozenClock(events))
        journal._events = events
        journal._seq = events[-1].seq if events else 0
        return journal


class _FrozenClock:
    """Clock stand-in for loaded journals: time never advances."""

    def __init__(self, events: list[Event]) -> None:
        self._last = events[-1].ts if events else 0.0

    def now(self) -> float:
        return self._last

    async def wait(self, dt: float) -> None:  # pragma: no cover - never awaited
        return None
