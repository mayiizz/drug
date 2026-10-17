"""
ASGI entry for: uvicorn app.main:app --app-dir backend

Ensures the repo root and ``backend`` are on ``sys.path`` so ``main`` and
``models.*`` imports resolve when the worker cwd is the project root.
"""
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[2]
_backend = Path(__file__).resolve().parents[1]
for _p in (_root, _backend):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from main import app

__all__ = ["app"]
