"""Lazy Neo4j Aura / Bolt driver helpers for CURA graph features."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_driver = None


class Neo4jNotConfiguredError(RuntimeError):
    """Raised when NEO4J_* env vars are missing or the driver cannot connect."""


def is_configured() -> bool:
    uri = (os.getenv("NEO4J_URI") or "").strip()
    user = (os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "").strip()
    password = (os.getenv("NEO4J_PASSWORD") or "").strip()
    return bool(uri and user and password)


def get_driver():
    """Return a shared neo4j.Driver, creating it on first use."""
    global _driver
    if _driver is not None:
        return _driver

    if not is_configured():
        raise Neo4jNotConfiguredError(
            "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, and "
            "NEO4J_PASSWORD in your .env (see .env.example), then run "
            "scripts/seed_neo4j.py."
        )

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise Neo4jNotConfiguredError(
            "neo4j Python package is not installed. Run: pip install neo4j"
        ) from exc

    uri = os.environ["NEO4J_URI"].strip()
    user = (os.getenv("NEO4J_USERNAME") or os.environ["NEO4J_USER"]).strip()
    password = os.environ["NEO4J_PASSWORD"].strip()

    _driver = GraphDatabase.driver(uri, auth=(user, password))
    logger.info("[Neo4j] Driver created for %s", uri)
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        try:
            _driver.close()
        except Exception as exc:
            logger.warning("[Neo4j] Driver close warning: %s", exc)
        _driver = None


def verify_connectivity() -> bool:
    """Return True if Neo4j responds; False if not configured or unreachable."""
    if not is_configured():
        return False
    try:
        driver = get_driver()
        driver.verify_connectivity()
        return True
    except Exception as exc:
        logger.warning("[Neo4j] Connectivity check failed: %s", exc)
        close_driver()  # clear bad pool after DNS / auth / pause failures
        return False


def run_cypher(
    query: str,
    params: dict[str, Any] | None = None,
    *,
    write: bool = False,
) -> list[dict[str, Any]]:
    """Execute a Cypher query and return a list of record dicts.

    Defaults to READ access (Aura-friendly for GraphRAG). Pass ``write=True``
    for MERGE/CREATE/constraints (seed script).
    """
    from neo4j import READ_ACCESS, WRITE_ACCESS

    driver = get_driver()
    params = params or {}
    access = WRITE_ACCESS if write else READ_ACCESS

    def _work(tx):
        result = tx.run(query, params)
        return [dict(record) for record in result]

    with driver.session(default_access_mode=access) as session:
        if write:
            return session.execute_write(_work)
        return session.execute_read(_work)
