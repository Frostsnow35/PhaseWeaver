"""Generate the self-contained Kaggle [L3] calibration notebook (v2).

The notebook is a build artifact (experiment carrier), so it is written to
``_local/`` which is gitignored per AGENTS.md section 1. Cell sources live in
``kaggle_cells.py``; this module is only the nbformat builder.

    python scripts/make_kaggle_notebook.py

Design constraint: Kaggle's 2x T4 is Turing sm_75 - no bf16, no FlashAttention-2,
and vLLM/SGLang multi-GPU on that pair is unreliable (NCCL / FlashInfer / libcuda
failures). Everything in the notebook therefore uses stock PyTorch + Transformers
on ONE card. No service, no NCCL, no pip install of an inference engine.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from kaggle_cells import CELLS, INTRO_LINES
    from kaggle_cells_late import LATE_CELLS
except ImportError:  # python -m scripts.make_kaggle_notebook
    from scripts.kaggle_cells import CELLS, INTRO_LINES
    from scripts.kaggle_cells_late import LATE_CELLS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_local" / "kaggle_l3_calibration.ipynb"


def _code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> int:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": INTRO_LINES}
        ] + [_code_cell(source) for source in [*CELLS, *LATE_CELLS]],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(CELLS) + len(LATE_CELLS)} code cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
