"""Statement coverage via the stdlib tracer (no third-party coverage dependency).

Why not coverage.py: this environment avoids adding dev dependencies that need a
network round-trip, and a line-hit tracer is enough to spot modules the tests never
touch. Numbers are approximations (statement start lines only), which is fine for
the audit - they are not used as a CI gate.

Run: python scripts/measure_coverage.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "src" / "phaseweaver").resolve()
sys.path.insert(0, str(ROOT / "src"))

hits: dict[str, set[int]] = {}


def tracer(frame, event, arg):  # type: ignore[no-untyped-def]
    if event == "line":
        filename = Path(frame.f_code.co_filename).resolve()
        try:
            filename.relative_to(SRC)
        except ValueError:
            return tracer
        hits.setdefault(str(filename), set()).add(frame.f_lineno)
    return tracer


def statements(path: Path) -> set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.lineno for node in ast.walk(tree) if isinstance(node, ast.stmt)}


def main() -> int:
    import pytest

    sys.settrace(tracer)
    try:
        pytest.main([str(ROOT / "tests"), "-q", "--no-header", "-p", "no:cacheprovider"])
    finally:
        sys.settrace(None)

    rows = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        total = statements(path)
        if not total:
            continue
        covered = len(total & hits.get(str(path), set()))
        rows.append((covered / len(total) * 100, covered, len(total), path))

    rows.sort()
    print(f"{'coverage':>9}  {'covered':>8}  {'stmts':>6}  module")
    for pct, covered, total, path in rows:
        print(f"{pct:8.1f}%  {covered:8d}  {total:6d}  {path.relative_to(ROOT)}")
    total_covered = sum(r[1] for r in rows)
    total_stmts = sum(r[2] for r in rows)
    print("-" * 70)
    print(f"overall: {total_covered / total_stmts * 100:.1f}%  ({total_covered}/{total_stmts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
