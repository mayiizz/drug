from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_CACHE: dict[str, Any] = {}


def _timed_singleton(name: str, factory):
    if name in _CACHE:
        return _CACHE[name]
    started = time.time()
    model = factory()
    _CACHE[name] = model
    logger.info("[Registry] Loaded %s in %.3fs", name, time.time() - started)
    return model


def get_deeppurpose_predictor():
    from models.binding_model import DeepPurposeBindingPredictor

    return _timed_singleton("deeppurpose_predictor", lambda: DeepPurposeBindingPredictor())


def get_safe_gpt_generator():
    from models.generation.safe_gpt_generator import ChemGPTGenerator

    return _timed_singleton("safe_gpt_generator", lambda: ChemGPTGenerator())


def get_chemberta_encoder():
    from models.embedding.embedding_engine import ChemBERTaEncoder

    return _timed_singleton("chemberta_encoder", lambda: ChemBERTaEncoder())


def get_esmfold_client():
    from models.esm_fold_client import ESMFoldClient

    return _timed_singleton("esmfold_client", lambda: ESMFoldClient())


def get_reinvent_generator():
    from models.reinvent_generator import ChemGPTGenerator

    return _timed_singleton("reinvent_generator", lambda: ChemGPTGenerator())


def preload_defaults() -> dict[str, bool]:
    statuses: dict[str, bool] = {}
    preloaders = {
        "safe_gpt_generator": get_safe_gpt_generator,
        "deeppurpose_predictor": get_deeppurpose_predictor,
        "chemberta_encoder": get_chemberta_encoder,
        "esmfold_client": get_esmfold_client,
    }
    for name, getter in preloaders.items():
        try:
            getter()
            statuses[name] = True
        except Exception as exc:
            statuses[name] = False
            logger.warning("[Registry] Preload failed for %s: %s", name, exc)
    return statuses

