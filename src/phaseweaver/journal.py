"""Append-only event journal: the single source of truth (ADR-0003)."""

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
