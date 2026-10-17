from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_DESIGNER_SINGLETON: object | None = None


def load_designer(verbose: bool = False) -> object:
    global _DESIGNER_SINGLETON
    if _DESIGNER_SINGLETON is not None:
        return _DESIGNER_SINGLETON

    logger.info("SAFE-GPT loader interpreter: %s", sys.executable)
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise ImportError(f"PyTorch is unavailable for SAFE-GPT runtime: {exc}") from exc
    try:
        import transformers  # noqa: F401
    except Exception as exc:
        raise ImportError(f"transformers is unavailable for SAFE-GPT runtime: {exc}") from exc

    sf = None
    try:
        # safe-mol package is typically imported as `safe`.
        sf = importlib.import_module("safe")
        logger.info("SAFE dependency module loaded from: %s", getattr(sf, "__file__", "<unknown>"))
    except Exception as exc_safe:
        # Best-effort visibility if user expects `safe_mol` module.
        try:
            sm = importlib.import_module("safe_mol")
            logger.info("safe_mol module loaded from: %s", getattr(sm, "__file__", "<unknown>"))
        except Exception:
            pass
        raise ImportError(
            "safe-mol runtime dependency is unavailable in the active interpreter. "
            f"Python executable: {sys.executable}. "
            "Activate the 'cura' conda environment and install safe-mol."
        ) from exc_safe

    try:
        _DESIGNER_SINGLETON = sf.SAFEDesign.load_default(verbose=verbose)  # type: ignore[union-attr]
        return _DESIGNER_SINGLETON
    except Exception as exc:
        logger.error("SAFE-GPT model load failed: %s", exc)
        raise

