from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def log_docking_event(event: str, payload: Dict[str, Any]) -> None:
    """
    Structured logging helper.

    Keep this lightweight so the pipelines can still run even if docking fails.
    """
    try:
        logger.warning("%s %s", event, json.dumps(payload, default=str))
    except Exception:
        logger.warning("%s (payload serialization failed)", event)


def log_docking_runtime(stage: str, start_time: float, extra: Optional[Dict[str, Any]] = None) -> None:
    dt = time.time() - start_time
    payload: Dict[str, Any] = {"stage": stage, "runtime_sec": round(dt, 3)}
    if extra:
        payload.update(extra)
    try:
        logger.info("docking_runtime %s", json.dumps(payload, default=str))
    except Exception:
        logger.info("docking_runtime %s", payload)

