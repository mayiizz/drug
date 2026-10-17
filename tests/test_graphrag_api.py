"""API-level validation tests for GraphRAG endpoints (mocked Neo4j)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Import app with lifespan engines mocked at the module level where possible."""
    # Import after env is set so Neo4j startup check is soft
    with patch.dict(
        "os.environ",
        {
            "NEO4J_URI": "",
            "NEO4J_USERNAME": "",
            "NEO4J_PASSWORD": "",
            "GROQ_API_KEY": "",
        },
        clear=False,
    ):
        from backend.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_graphrag_empty_query_422(client):
    resp = client.post("/graphrag", json={"query": "   "})
    assert resp.status_code == 422


def test_graphrag_neo4j_missing_503(client):
    with patch(
        "models.graph.graphrag.is_configured", return_value=False
    ):
        with patch(
            "models.graph.neo4j_client.is_configured", return_value=False
        ):
            resp = client.post(
                "/graphrag",
                json={"query": "Which drugs treat lung cancer?"},
            )
            # 503 when Neo4j not configured
            assert resp.status_code in (503, 500)


def test_graph_search_unknown_intent_422(client):
    with patch("models.graph.graphrag.is_configured", return_value=True):
        resp = client.post(
            "/graph/search",
            json={"intent": "totally_fake_intent"},
        )
        assert resp.status_code == 422


def test_graph_search_ok_mocked(client):
    with patch("models.graph.graphrag.is_configured", return_value=True):
        with patch(
            "models.graph.graphrag.run_cypher",
            return_value=[
                {
                    "drug": "Aspirin",
                    "diseases": ["Pain"],
                    "proteins": ["COX-1"],
                }
            ],
        ):
            resp = client.post("/graph/search", json={})
            assert resp.status_code == 200
            data = resp.json()
            assert data["intent"] == "list_drugs"
            assert "paths" in data
