"""Unit tests for GraphRAG routing, retrieval formatting, and fallbacks.

External Neo4j / Groq are mocked — no live credentials required.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from models.graph import graphrag as gr
from models.graph.neo4j_client import Neo4jNotConfiguredError


@pytest.fixture(autouse=True)
def _clear_entity_cache(monkeypatch):
    gr._ENTITY_CACHE["drugs"] = {
        "erlotinib",
        "gefitinib",
        "metformin",
        "imatinib",
    }
    gr._ENTITY_CACHE["diseases"] = {"lung cancer", "leukemia", "type 2 diabetes"}
    gr._ENTITY_CACHE["proteins"] = {"egfr", "bcr-abl", "kras"}
    gr._ENTITY_CACHE["side_effects"] = {"nausea", "rash", "diarrhea"}
    gr._ENTITY_CACHE["concepts"] = {"egfr inhibition"}
    gr._ENTITY_CACHE["methods"] = {"egfr tki therapy"}
    gr._ENTITY_CACHE["limitations"] = {"skin toxicity / rash"}
    gr._ENTITY_CACHE["authors"] = {"shepherd fa"}
    gr._ENTITY_CACHE_TS = 1e12  # keep cache warm

    def _canon(matched_lower: str | None, label: str) -> str | None:
        if not matched_lower:
            return None
        special = {
            "egfr": "EGFR",
            "bcr-abl": "BCR-ABL",
            "kras": "KRAS",
            "lung cancer": "Lung cancer",
            "type 2 diabetes": "Type 2 diabetes",
            "erlotinib": "Erlotinib",
            "gefitinib": "Gefitinib",
            "metformin": "Metformin",
            "imatinib": "Imatinib",
            "nausea": "Nausea",
            "rash": "Rash",
            "diarrhea": "Diarrhea",
            "leukemia": "Leukemia",
            "egfr inhibition": "EGFR inhibition",
            "egfr tki therapy": "EGFR TKI therapy",
            "skin toxicity / rash": "Skin toxicity / rash",
            "shepherd fa": "Shepherd FA",
        }
        return special.get(matched_lower, matched_lower.title())

    monkeypatch.setattr(gr, "_canonical_name", _canon)
    monkeypatch.setattr(gr, "_refresh_entity_cache", lambda force=False: None)
    yield
    gr._ENTITY_CACHE_TS = 0.0


def test_route_intent_multi_hop_egfr():
    intent, params, warning = gr._route_intent(
        "Which drugs target EGFR, what disease relationships are recorded, "
        "and what side effects are associated?"
    )
    assert intent == "multi_hop_research"
    assert params.get("protein") == "EGFR"
    assert warning is None


def test_route_intent_research_discovery():
    intent, params, warning = gr._route_intent(
        "Which approaches target EGFR, what limitations are recorded, "
        "and which papers discuss related drugs?"
    )
    assert intent == "research_discovery"
    assert params.get("protein") == "EGFR"
    assert warning is None


def test_route_intent_research_discovery_missing_entity():
    intent, params, warning = gr._route_intent(
        "Which approaches and limitations are recorded in the research graph?"
    )
    assert intent == "research_discovery"
    assert "protein" not in params
    assert "concept" not in params
    assert warning and "missing_required_entity" in warning


def test_format_retrieval_research_discovery_paths():
    records = [
        {
            "method": "EGFR TKI therapy",
            "protein": "EGFR",
            "limitations": ["Skin toxicity / rash", "GI toxicity / diarrhea"],
            "side_effects": ["Rash", "Diarrhea"],
            "concepts": ["EGFR inhibition"],
            "authors": ["Shepherd FA", "Lynch TJ"],
            "related_drugs": ["Erlotinib"],
            "documents": ["EGFR TKI approaches overview"],
            "papers": [
                {
                    "pmid": "15118073",
                    "title": "Erlotinib in NSCLC",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/15118073/",
                    "year": 2005,
                }
            ],
            "related_papers": [
                {
                    "pmid": "16014882",
                    "title": "EGFR mutations",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/16014882/",
                    "year": 2004,
                }
            ],
        }
    ]
    out = gr._format_retrieval("research_discovery", {"protein": "EGFR"}, records)
    assert out["evidence_status"] == "ok"
    assert any("PROPOSES" in p and "HAS_LIMITATION" in p for p in out["paths"])
    assert any("TARGETS" in p for p in out["paths"])
    assert any("RELATED_TO" in p for p in out["paths"])
    assert "Shepherd FA" in out["authors"]
    assert out["papers"][0]["pmid"] == "15118073"


def test_route_intent_shared_targets_requires_drug():
    intent, params, warning = gr._route_intent(
        "Which drugs share a common target?"
    )
    assert intent == "shared_targets"
    assert "drug" not in params
    assert warning and warning.startswith("missing_required_entity")


def test_route_intent_side_effects_no_silent_erlotinib():
    intent, params, warning = gr._route_intent("What are the side effects?")
    assert intent == "side_effects"
    assert params.get("drug") != "Erlotinib"
    assert warning and "missing_required_entity" in warning


def test_route_intent_treats_disease():
    intent, params, warning = gr._route_intent("Which drugs treat lung cancer?")
    assert intent == "treats_disease"
    assert params.get("disease") == "Lung cancer"
    assert warning is None


def test_route_intent_similar_with_drug():
    intent, params, warning = gr._route_intent("Drugs similar to Gefitinib")
    assert intent == "similar_drugs"
    assert params.get("drug") == "Gefitinib"
    assert warning is None


def test_format_retrieval_multi_hop_paths():
    records = [
        {
            "drug": "Erlotinib",
            "protein": "EGFR",
            "treated_diseases": ["Lung cancer"],
            "associated_diseases": ["Lung cancer"],
            "side_effects": ["Rash", "Diarrhea"],
            "similar_drugs": ["Gefitinib"],
            "papers": [],
        }
    ]
    out = gr._format_retrieval("multi_hop_research", {"protein": "EGFR"}, records)
    assert out["evidence_status"] == "ok"
    assert any("BINDS" in p for p in out["paths"])
    assert any("TREATS" in p for p in out["paths"])
    assert any("HAS_SIDE_EFFECT" in p for p in out["paths"])
    assert "Erlotinib" in out["entities"]
    assert "EGFR" in out["entities"]


def test_format_retrieval_empty():
    out = gr._format_retrieval("treats_disease", {"disease": "XYZ"}, [])
    assert out["paths"] == []
    assert out["evidence_status"] == "none"


def test_retrieve_missing_neo4j():
    with patch.object(gr, "is_configured", return_value=False):
        with pytest.raises(Neo4jNotConfiguredError):
            gr.retrieve("Which drugs treat lung cancer?")


def test_retrieve_missing_entity_no_cypher_default():
    with patch.object(gr, "is_configured", return_value=True):
        with patch.object(gr, "run_cypher") as mock_cypher:
            out = gr.retrieve("What are the side effects?")
            mock_cypher.assert_not_called()
            assert out["evidence_status"] == "none"
            assert out["paths"] == []
            assert "missing_required_entity" in (out.get("warning") or "")


def test_retrieve_single_hop_mocked():
    with patch.object(gr, "is_configured", return_value=True):
        with patch.object(
            gr,
            "run_cypher",
            return_value=[
                {
                    "drug": "Erlotinib",
                    "disease": "Lung cancer",
                    "proteins": ["EGFR"],
                    "side_effects": ["Rash"],
                }
            ],
        ):
            out = gr.retrieve("Which drugs treat lung cancer?")
            assert out["intent"] == "treats_disease"
            assert out["paths"]
            assert out["evidence_status"] == "ok"


def test_retrieve_multi_hop_mocked():
    with patch.object(gr, "is_configured", return_value=True):
        with patch.object(
            gr,
            "run_cypher",
            return_value=[
                {
                    "drug": "Erlotinib",
                    "protein": "EGFR",
                    "treated_diseases": ["Lung cancer"],
                    "associated_diseases": ["Lung cancer"],
                    "side_effects": ["Rash"],
                    "similar_drugs": [],
                    "papers": [
                        {
                            "pmid": "15118073",
                            "title": "Erlotinib in NSCLC",
                            "url": "https://pubmed.ncbi.nlm.nih.gov/15118073/",
                            "year": 2005,
                        }
                    ],
                }
            ],
        ):
            out = gr.retrieve(
                "Which drugs target EGFR and what side effects are associated?"
            )
            assert out["intent"] == "multi_hop_research"
            assert out["papers"]
            assert out["papers"][0]["pmid"] == "15118073"


def test_answer_no_evidence_skips_groq():
    with patch.object(gr, "is_configured", return_value=True):
        with patch.object(
            gr,
            "retrieve",
            return_value={
                "intent": "side_effects",
                "params": {},
                "paths": [],
                "entities": [],
                "papers": [],
                "evidence_status": "none",
                "warning": "missing_required_entity: name a drug",
            },
        ):
            with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}):
                # Even with a key, empty paths must not invent facts
                out = gr.answer("What are the side effects?")
                assert out["evidence_status"] == "none"
                assert "No graph evidence" in out["answer"] or "No matching" in out["answer"]


def test_answer_paths_only_without_groq():
    with patch.object(gr, "is_configured", return_value=True):
        with patch.object(
            gr,
            "retrieve",
            return_value={
                "intent": "treats_disease",
                "params": {"disease": "Lung cancer"},
                "paths": ["Erlotinib -TREATS-> Lung cancer"],
                "entities": ["Erlotinib", "Lung cancer"],
                "papers": [],
                "evidence_status": "ok",
            },
        ):
            env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                out = gr.answer("Which drugs treat lung cancer?")
                assert out["evidence_status"] == "paths_only"
                assert "Erlotinib -TREATS-> Lung cancer" in out["answer"]


def test_search_unknown_intent():
    with patch.object(gr, "is_configured", return_value=True):
        with pytest.raises(ValueError, match="Unknown intent"):
            gr.search(intent="not_a_real_intent")


def test_search_missing_required_param():
    with patch.object(gr, "is_configured", return_value=True):
        out = gr.search(intent="similar_drugs")  # no drug
        assert out["evidence_status"] == "none"
        assert "missing_required_entity" in (out.get("warning") or "")


def test_search_list_drugs_mocked():
    with patch.object(gr, "is_configured", return_value=True):
        with patch.object(
            gr,
            "run_cypher",
            return_value=[
                {"drug": "Aspirin", "diseases": ["Pain"], "proteins": ["COX-1"]}
            ],
        ):
            out = gr.search()
            assert out["intent"] == "list_drugs"
            assert out["paths"]
