"""Shared modules for the three emotion-recognition pipelines.

Importing this package makes the repository root importable, so a pipeline can be
run directly as a script (``python models/speech_pipeline/train.py``) without
installing the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
