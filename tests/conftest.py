"""Test bootstrap.

Adds the ``src`` layout and the ``tests`` directory to ``sys.path`` so the
``reel_af`` package and the local ``util`` helper module import cleanly
whether or not the project has been ``pip install``-ed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_TESTS = Path(__file__).resolve().parent

for _p in (str(_SRC), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
