"""Phase timelines: turn the event log into a Gantt-friendly structure.

The log is already the source of truth; this module only *projects* it. There is no
runtime overhead because nothing is recorded here - a timeline is built by replaying
events, which is also why it can be produced for a run that already finished.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from phaseweaver.journal import EventJournal
from phaseweaver.types import Phase


@dataclass(frozen=True)
class Span:
    unit_id: str
    kind: str  # "phase" | "context"
    name: str  # phase name or context state
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Timeline:
    run_id: str
    wall_clock_s: float
    spans: list[Span] = field(default_factory=list)

    def for_unit(self, unit_id: str) -> list[Span]:
        return [span for span in self.spans if span.unit_id == unit_id]

    def total(self, unit_id: str, phase: str) -> float:
        return sum(
            span.duration
            for span in self.spans
            if span.unit_id == unit_id and span.kind == "phase" and span.name == phase
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "wall_clock_s": self.wall_clock_s,
            "spans": [
                {
                    "unit_id": s.unit_id,
                    "kind": s.kind,
                    "name": s.name,
                    "start": s.start,
                    "end": s.end,
                    "duration": s.duration,
                }
                for s in self.spans
            ],
        }

    def to_jsonl(self, path: Path) -> None:
        """One JSON object per unit - convenient for notebook / Gantt tooling."""
        by_unit: dict[str, list[Span]] = {}
        for span in self.spans:
            by_unit.setdefault(span.unit_id, []).append(span)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for unit_id, spans in by_unit.items():
                fh.write(
                    json.dumps(
                        {
                            "unit_id": unit_id,
                            "spans": [
                                {
                                    "kind": s.kind,
                                    "name": s.name,
                                    "start": s.start,
                                    "end": s.end,
                                }
                                for s in spans
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def build_timeline(journal: EventJournal) -> Timeline:
    """Replay phase and context transitions into closed intervals."""
    end_of_run = 0.0
    for event in journal.events():
        end_of_run = max(end_of_run, event.ts)

    open_spans: dict[tuple[str, str], Span] = {}
    spans: list[Span] = []

    def close(unit_id: str, kind: str, at: float) -> None:
        key = (unit_id, kind)
        current = open_spans.pop(key, None)
        if current is None:
            return
        spans.append(Span(current.unit_id, current.kind, current.name, current.start, at))

    for event in journal.events():
        if event.unit_id is None:
            continue
        if event.event == "phase_transitioned":
            to = str(event.payload.get("to", ""))
            if not to:
                continue
            close(event.unit_id, "phase", event.ts)
            if not Phase(to).is_terminal():
                open_spans[(event.unit_id, "phase")] = Span(
                    event.unit_id, "phase", to, event.ts, event.ts
                )
        elif event.event == "context_state_changed":
            state = str(event.payload.get("state", ""))
            if not state:
                continue
            close(event.unit_id, "context", event.ts)
            open_spans[(event.unit_id, "context")] = Span(
                event.unit_id, "context", state, event.ts, event.ts
            )

    for current in open_spans.values():
        spans.append(Span(current.unit_id, current.kind, current.name, current.start, end_of_run))
    spans.sort(key=lambda s: (s.start, s.unit_id, s.kind))
    return Timeline(run_id=journal.run_id, wall_clock_s=end_of_run, spans=spans)
