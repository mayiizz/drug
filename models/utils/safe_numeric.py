"""Numeric coercion helpers safe for FastAPI / JSON serialisation.

The CURA pipeline produces a mix of plain floats, NumPy scalars, pandas
Series elements, and ``None`` from optional models (DeepPurpose, Vina,
Tox21, ADMET). Forwarding ``NaN`` / ``Inf`` straight to FastAPI causes
JSON serialisation failures (``ValueError: Out of range float values are
not JSON compliant``). Use :func:`safe_float` at the response boundary so
the API never blows up on a missing or pathological metric.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["safe_float", "safe_int", "safe_round"]


def safe_float(x: Any, default: float | None = None) -> float | None:
    """Coerce ``x`` to a finite ``float`` or return ``default``.

    Handles ``None``, NumPy scalars (via ``.item()`` fallback), strings
    parseable as numbers, and rejects ``NaN`` / ``+/-Inf``.
    """
    if x is None:
        return default
    # NumPy scalars expose ``.item()``; pandas Series elements that survive a
    # ``.iloc[0]`` already are scalars but may still be numpy types.
    item = getattr(x, "item", None)
    if callable(item):
        try:
            x = item()
        except Exception:
            pass
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def safe_int(x: Any, default: int | None = 0) -> int | None:
    """Coerce ``x`` to ``int`` or return ``default`` on failure.

    Truncates via ``int(float(...))`` so ``"3.0"`` becomes ``3``.
    """
    if x is None:
        return default
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def safe_round(x: Any, ndigits: int = 2, default: float | None = None) -> float | None:
    """Round a numeric-ish input safely. Returns ``default`` if not numeric."""
    v = safe_float(x, default=None)
    if v is None:
        return default
    try:
        return round(v, ndigits)
    except Exception:
        return default
