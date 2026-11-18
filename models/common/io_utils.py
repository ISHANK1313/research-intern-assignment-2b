"""Tiny JSON helpers with no heavy dependencies.

Kept separate from ``train_utils`` on purpose: the plotting, summarisation and report
scripts only need to read and write JSON, and importing ``train_utils`` would drag in
PyTorch. Keeping this path torch-free means results can be re-plotted and the report
rebuilt even on a machine where PyTorch is unavailable or misconfigured.
"""

from __future__ import annotations

import json
from pathlib import Path


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
