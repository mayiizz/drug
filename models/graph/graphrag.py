"""GraphRAG: multi-hop Cypher retrieval from Neo4j + Groq synthesis."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

from models.graph.neo4j_client import Neo4jNotConfiguredError, is_configured, run_cypher

load_dotenv()

logger = logging.getLogger(__name__)

MAX_PATHS = 40
MAX_EVIDENCE_CHARS = 6000

# Cache of entity names loaded from Neo4j (refreshed periodically).
_ENTITY_CACHE: dict[str, set[str]] = {
    "drugs": set(),
    "diseases": set(),
    "proteins": set(),
    "side_effects": set(),
    "concepts": set(),
    "methods": set(),
    "limitations": set(),
    "authors": set(),
}
_ENTITY_CACHE_TS: float = 0.0
_ENTITY_CACHE_TTL = 300.0  # seconds

# Intents that require a named entity parameter.
_REQUIRED_PARAM: dict[str, str] = {
    "drug_neighborhood": "drug",
    "shared_targets": "drug",
    "similar_drugs": "drug",
    "side_effects": "drug",
    "disease_neighborhood": "disease",
    "treats_disease": "disease",
    "protein_neighborhood": "protein",
    "multi_hop_research": "protein",
    "drugs_by_side_effect": "side_effect",
    "papers_for_entity": "entity",
}


def _refresh_entity_cache(force: bool = False) -> None:
    """Load entity names from Neo4j into memory (incl. research subgraph)."""
    global _ENTITY_CACHE_TS
    now = time.time()
    if (
        not force
        and _ENTITY_CACHE_TS
        and (now - _ENTITY_CACHE_TS) < _ENTITY_CACHE_TTL
        and any(_ENTITY_CACHE.values())
    ):
        return
    if not is_configured():
        return
    try:
        # Explicit per-label collects (avoid labels(n)[0] ambiguity).
        drugs = {
            str(r["n"]).lower()
            for r in run_cypher("MATCH (d:Drug) RETURN toLower(d.name) AS n")
            if r.get("n")
        }
        diseases = {
            str(r["n"]).lower()
            for r in run_cypher("MATCH (d:Disease) RETURN toLower(d.name) AS n")
            if r.get("n")
        }
        proteins = {
            str(r["n"]).lower()
            for r in run_cypher("MATCH (p:Protein) RETURN toLower(p.name) AS n")
            if r.get("n")
        }
        side_effects = {
            str(r["n"]).lower()
            for r in run_cypher("MATCH (s:SideEffect) RETURN toLower(s.name) AS n")
            if r.get("n")
        }
        concepts = {
            str(r["n"]).lower()
            for r in run_cypher("MATCH (c:Concept) RETURN toLower(c.name) AS n")
            if r.get("n")
        }
        methods = {
            str(r["n"]).lower()
            for r in run_cypher("MATCH (m:Method) RETURN toLower(m.name) AS n")
            if r.get("n")
        }
        limitations = {
            str(r["n"]).lower()
            for r in run_cypher("MATCH (l:Limitation) RETURN toLower(l.name) AS n")
            if r.get("n")
        }
        authors = {
            str(r["n"]).lower()
            for r in run_cypher("MATCH (a:Author) RETURN toLower(a.name) AS n")
            if r.get("n")
        }
        _ENTITY_CACHE["drugs"] = drugs
        _ENTITY_CACHE["diseases"] = diseases
        _ENTITY_CACHE["proteins"] = proteins
        _ENTITY_CACHE["side_effects"] = side_effects
        _ENTITY_CACHE["concepts"] = concepts
        _ENTITY_CACHE["methods"] = methods
        _ENTITY_CACHE["limitations"] = limitations
        _ENTITY_CACHE["authors"] = authors
        _ENTITY_CACHE_TS = now
        logger.info(
            "[GraphRAG] Entity cache: drugs=%d diseases=%d proteins=%d "
            "side_effects=%d concepts=%d methods=%d limitations=%d authors=%d",
            len(drugs),
            len(diseases),
            len(proteins),
            len(side_effects),
            len(concepts),
            len(methods),
            len(limitations),
            len(authors),
        )
    except Exception as exc:
        logger.warning("[GraphRAG] Entity cache refresh failed: %s", exc)


def _extract_entity(query: str, candidates: set[str]) -> str | None:
    """Longest case-insensitive substring match against candidate names."""
    q = query.lower()
    for name in sorted(candidates, key=len, reverse=True):
        if name and name in q:
            return name
    return None


def _canonical_name(matched_lower: str | None, label: str) -> str | None:
    """Resolve a lowercased match back to the graph's stored casing."""
    if not matched_lower:
        return None
    try:
        rows = run_cypher(
            f"MATCH (n:{label}) WHERE toLower(n.name) = $name RETURN n.name AS name LIMIT 1",
            {"name": matched_lower},
        )
        if rows:
            return rows[0]["name"]
    except Exception:
        pass
    if matched_lower.isupper() or len(matched_lower) <= 6:
        return matched_lower.upper()
    return matched_lower.title()


def _empty_retrieval(
    intent: str,
    params: dict[str, Any],
    *,
    warning: str,
) -> dict[str, Any]:
    return {
        "intent": intent,
        "params": params,
        "records": [],
        "paths": [],
        "entities": [],
        "papers": [],
        "authors": [],
        "warning": warning,
        "evidence_status": "none",
    }


def _route_intent(query: str) -> tuple[str, dict[str, Any], str | None]:
    """Return (intent, params, warning). Warning set when required entity missing."""
    _refresh_entity_cache()
    q = query.lower()

    drug = _canonical_name(_extract_entity(q, _ENTITY_CACHE["drugs"]), "Drug")
    disease = _canonical_name(_extract_entity(q, _ENTITY_CACHE["diseases"]), "Disease")
    protein = _canonical_name(_extract_entity(q, _ENTITY_CACHE["proteins"]), "Protein")
    side_effect = _canonical_name(
        _extract_entity(q, _ENTITY_CACHE["side_effects"]), "SideEffect"
    )
    concept = _canonical_name(
        _extract_entity(q, _ENTITY_CACHE["concepts"]), "Concept"
    )

    entity_types = sum(
        1 for x in (drug, disease, protein, side_effect) if x is not None
    )
    multi_hop_cue = any(
        w in q
        for w in (
            "multi-hop",
            "multi hop",
            "and what",
            "and which",
            "side effect",
            "disease relationship",
            "relationships",
            "associated with",
        )
    )
    # Track-shaped research subgraph: approaches / methods / limitations / related papers
    research_cue = any(
        w in q
        for w in (
            "approach",
            "approaches",
            "limitation",
            "limitations",
            "papers discuss",
            "related paper",
            "related papers",
            "research discovery",
            "authored",
            "authors of",
            "proposes",
            "has_limitation",
            "what methods",
            "which methods",
        )
    )

    # Research knowledge discovery (Paper→Author→Concept→Method→Limitation)
    if research_cue:
        params: dict[str, Any] = {}
        if protein:
            params["protein"] = protein
        if concept:
            params["concept"] = concept
        if params:
            return "research_discovery", params, None
        return (
            "research_discovery",
            {},
            "missing_required_entity: name a protein or concept in the research "
            "subgraph (e.g. EGFR, EGFR inhibition)",
        )

    # Paper / literature questions over the curated Paper subgraph
    if any(w in q for w in ("paper", "papers", "pubmed", "literature", "publication", "cite")):
        ent = drug or protein or disease or concept
        if ent:
            return "papers_for_entity", {"entity": ent}, None
        return (
            "papers_for_entity",
            {},
            "missing_required_entity: name a drug, protein, or disease for paper lookup",
        )

    # Compositional multi-hop: protein + (disease / side effects / multi cues / 2+ entities)
    if protein and (
        disease
        or side_effect
        or multi_hop_cue
        or entity_types >= 2
        or ("target" in q and ("disease" in q or "side" in q))
    ):
        params = {"protein": protein}
        if disease:
            params["disease"] = disease
        return "multi_hop_research", params, None

    if protein and disease:
        return "multi_hop_research", {"protein": protein, "disease": disease}, None

    # Catalog / list questions
    if any(
        phrase in q
        for phrase in (
            "list drugs",
            "all drugs",
            "what drugs",
            "which drugs are",
            "show drugs",
            "catalog",
            "how many drugs",
        )
    ) and not drug and not disease and not protein and not side_effect:
        return "list_drugs", {}, None

    if any(w in q for w in ("side effect", "adverse", "toxicity", "adr")):
        if side_effect and not drug:
            return "drugs_by_side_effect", {"side_effect": side_effect}, None
        if not drug:
            return (
                "side_effects",
                {},
                "missing_required_entity: name a drug in the seeded graph "
                "(e.g. Erlotinib) for side-effect lookup",
            )
        return "side_effects", {"drug": drug}, None

    if side_effect and any(w in q for w in ("cause", "causes", "with", "have", "has")):
        return "drugs_by_side_effect", {"side_effect": side_effect}, None

    if any(w in q for w in ("similar", "analog", "like")):
        if not drug:
            return (
                "similar_drugs",
                {},
                "missing_required_entity: name a drug in the seeded graph for similarity",
            )
        return "similar_drugs", {"drug": drug}, None

    if any(w in q for w in ("share", "common target", "same target", "shared")):
        if not drug:
            return (
                "shared_targets",
                {},
                "missing_required_entity: name a drug in the seeded graph "
                "for shared-target lookup",
            )
        return "shared_targets", {"drug": drug}, None

    if any(w in q for w in ("treat", "treats", "treatment", "indication", "therapy")):
        if disease:
            return "treats_disease", {"disease": disease}, None
        if drug:
            return "drug_neighborhood", {"drug": drug}, None
        return (
            "treats_disease",
            {},
            "missing_required_entity: name a disease in the seeded graph "
            "(e.g. Lung cancer)",
        )

    if protein and ("drug" in q or "inhibit" in q or "bind" in q or "target" in q):
        return "protein_neighborhood", {"protein": protein}, None

    if disease and not drug:
        if any(w in q for w in ("drug", "treat", "therapy", "medicine")):
            return "treats_disease", {"disease": disease}, None
        return "disease_neighborhood", {"disease": disease}, None

    if drug:
        return "drug_neighborhood", {"drug": drug}, None

    if disease:
        return "treats_disease", {"disease": disease}, None

    if protein:
        return "protein_neighborhood", {"protein": protein}, None

    if side_effect:
        return "drugs_by_side_effect", {"side_effect": side_effect}, None

    # Free-text: Capitalized token that looks like a drug name
    m = re.search(r"\b([A-Z][a-z]+(?:ib|inib|mab|tinib|ciclib|statin|olol|pril)?)\b", query)
    if m:
        guessed = m.group(1)
        # Only accept if present in graph (case-insensitive)
        if guessed.lower() in _ENTITY_CACHE["drugs"]:
            canon = _canonical_name(guessed.lower(), "Drug")
            return "drug_neighborhood", {"drug": canon or guessed}, None
        return (
            "drug_neighborhood",
            {"drug": guessed},
            f"unknown_entity: '{guessed}' is not in the seeded Drug graph",
        )

    if any(w in q for w in ("overview", "summary", "graph", "knowledge")):
        return "overview", {}, None

    return "list_drugs", {}, None


_QUERIES: dict[str, str] = {
    "drug_neighborhood": """
        MATCH (d:Drug)
        WHERE toLower(d.name) = toLower($drug)
        OPTIONAL MATCH (d)-[:BINDS]->(p:Protein)
        OPTIONAL MATCH (d)-[:TREATS]->(dis:Disease)
        OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(adis:Disease)
        OPTIONAL MATCH (d)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        OPTIONAL MATCH (d)-[sim:SIMILAR_TO]->(other:Drug)
        OPTIONAL MATCH (paper:Paper)-[:DISCUSSES]->(d)
        RETURN d.name AS drug,
               collect(DISTINCT p.name)[0..12] AS proteins,
               collect(DISTINCT dis.name)[0..12] AS diseases,
               collect(DISTINCT adis.name)[0..8] AS associated_diseases,
               collect(DISTINCT s.name)[0..10] AS side_effects,
               collect(DISTINCT other.name)[0..10] AS similar_drugs,
               collect(DISTINCT {
                 pmid: paper.pmid, title: paper.title,
                 url: paper.url, year: paper.year
               })[0..6] AS papers
        LIMIT 5
    """,
    "disease_neighborhood": """
        MATCH (dis:Disease)
        WHERE toLower(dis.name) = toLower($disease)
        OPTIONAL MATCH (d:Drug)-[:TREATS]->(dis)
        OPTIONAL MATCH (p:Protein)-[:ASSOCIATED_WITH]->(dis)
        OPTIONAL MATCH (d2:Drug)-[:BINDS]->(p)
        OPTIONAL MATCH (paper:Paper)-[:DISCUSSES]->(dis)
        RETURN dis.name AS disease,
               collect(DISTINCT d.name)[0..20] AS treating_drugs,
               collect(DISTINCT p.name)[0..15] AS proteins,
               collect(DISTINCT d2.name)[0..20] AS binder_drugs,
               collect(DISTINCT {
                 pmid: paper.pmid, title: paper.title,
                 url: paper.url, year: paper.year
               })[0..6] AS papers
        LIMIT 5
    """,
    "treats_disease": """
        MATCH (d:Drug)-[:TREATS]->(dis:Disease)
        WHERE toLower(dis.name) = toLower($disease)
        OPTIONAL MATCH (d)-[:BINDS]->(p:Protein)
        OPTIONAL MATCH (d)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        RETURN d.name AS drug, dis.name AS disease,
               collect(DISTINCT p.name) AS proteins,
               collect(DISTINCT s.name)[0..6] AS side_effects
        ORDER BY d.name
        LIMIT 40
    """,
    "protein_neighborhood": """
        MATCH (d:Drug)-[:BINDS]->(p:Protein)
        WHERE toLower(p.name) = toLower($protein)
        OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(dis:Disease)
        OPTIONAL MATCH (d)-[:TREATS]->(td:Disease)
        OPTIONAL MATCH (paper:Paper)-[:DISCUSSES]->(p)
        RETURN d.name AS drug, p.name AS protein,
               collect(DISTINCT dis.name) AS associated_diseases,
               collect(DISTINCT td.name) AS treated_diseases,
               collect(DISTINCT {
                 pmid: paper.pmid, title: paper.title,
                 url: paper.url, year: paper.year
               })[0..6] AS papers
        LIMIT 40
    """,
    "multi_hop_research": """
        MATCH (d:Drug)-[:BINDS]->(p:Protein)
        WHERE toLower(p.name) = toLower($protein)
        OPTIONAL MATCH (d)-[:TREATS]->(td:Disease)
        OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(adis:Disease)
        OPTIONAL MATCH (d)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        OPTIONAL MATCH (d)-[sim:SIMILAR_TO]->(other:Drug)
        OPTIONAL MATCH (paper:Paper)-[:DISCUSSES]->(p)
        OPTIONAL MATCH (paper2:Paper)-[:DISCUSSES]->(d)
        RETURN d.name AS drug, p.name AS protein,
               collect(DISTINCT td.name) AS treated_diseases,
               collect(DISTINCT adis.name) AS associated_diseases,
               collect(DISTINCT s.name)[0..8] AS side_effects,
               collect(DISTINCT other.name)[0..6] AS similar_drugs,
               collect(DISTINCT {
                 pmid: coalesce(paper.pmid, paper2.pmid),
                 title: coalesce(paper.title, paper2.title),
                 url: coalesce(paper.url, paper2.url),
                 year: coalesce(paper.year, paper2.year)
               })[0..8] AS papers
        ORDER BY d.name
        LIMIT 40
    """,
    "shared_targets": """
        MATCH (d:Drug)-[:BINDS]->(p:Protein)<-[:BINDS]-(other:Drug)
        WHERE toLower(d.name) = toLower($drug) AND other <> d
        OPTIONAL MATCH (other)-[:TREATS]->(dis:Disease)
        RETURN d.name AS drug, p.name AS protein, other.name AS related_drug,
               collect(DISTINCT dis.name) AS diseases
        LIMIT 30
    """,
    "similar_drugs": """
        MATCH (d:Drug)-[sim:SIMILAR_TO]->(other:Drug)
        WHERE toLower(d.name) = toLower($drug)
        OPTIONAL MATCH (other)-[:BINDS]->(p:Protein)
        OPTIONAL MATCH (other)-[:TREATS]->(dis:Disease)
        RETURN d.name AS drug, other.name AS similar_drug, sim.score AS score,
               collect(DISTINCT p.name) AS proteins,
               collect(DISTINCT dis.name) AS diseases
        ORDER BY score DESC
        LIMIT 20
    """,
    "side_effects": """
        MATCH (d:Drug)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        WHERE toLower(d.name) = toLower($drug)
        OPTIONAL MATCH (d)-[:BINDS]->(p:Protein)
        OPTIONAL MATCH (d)-[:TREATS]->(dis:Disease)
        RETURN d.name AS drug, collect(DISTINCT s.name) AS side_effects,
               collect(DISTINCT p.name) AS proteins,
               collect(DISTINCT dis.name) AS diseases
        LIMIT 10
    """,
    "drugs_by_side_effect": """
        MATCH (d:Drug)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        WHERE toLower(s.name) = toLower($side_effect)
        OPTIONAL MATCH (d)-[:BINDS]->(p:Protein)
        OPTIONAL MATCH (d)-[:TREATS]->(dis:Disease)
        RETURN s.name AS side_effect, d.name AS drug,
               collect(DISTINCT p.name) AS proteins,
               collect(DISTINCT dis.name) AS diseases
        ORDER BY d.name
        LIMIT 40
    """,
    "list_drugs": """
        MATCH (d:Drug)
        OPTIONAL MATCH (d)-[:TREATS]->(dis:Disease)
        OPTIONAL MATCH (d)-[:BINDS]->(p:Protein)
        RETURN d.name AS drug,
               collect(DISTINCT dis.name)[0..5] AS diseases,
               collect(DISTINCT p.name)[0..5] AS proteins
        ORDER BY d.name
        LIMIT 80
    """,
    "overview": """
        MATCH (d:Drug)-[:BINDS]->(p:Protein)
        OPTIONAL MATCH (d)-[:TREATS]->(dis:Disease)
        RETURN d.name AS drug, p.name AS protein, dis.name AS disease
        LIMIT 50
    """,
    "papers_for_entity": """
        MATCH (paper:Paper)-[:DISCUSSES]->(n)
        WHERE toLower(coalesce(n.name, '')) = toLower($entity)
           OR toLower(coalesce(n.name, '')) CONTAINS toLower($entity)
        RETURN paper.pmid AS pmid, paper.title AS title,
               paper.url AS url, paper.year AS year,
               labels(n)[0] AS entity_label, n.name AS entity_name
        ORDER BY paper.year DESC
        LIMIT 20
    """,
    "research_discovery": """
        MATCH (m:Method)-[:TARGETS]->(p:Protein)
        WHERE ($protein IS NOT NULL AND toLower(p.name) = toLower($protein))
           OR ($concept IS NOT NULL AND EXISTS {
                MATCH (paper0:Paper)-[:PROPOSES]->(m)
                MATCH (paper0)-[:DISCUSSES]->(c0:Concept)
                WHERE toLower(c0.name) = toLower($concept)
                   OR toLower(c0.name) CONTAINS toLower($concept)
           })
        OPTIONAL MATCH (m)-[:HAS_LIMITATION]->(l:Limitation)
        OPTIONAL MATCH (l)-[:MANIFESTS_AS]->(se:SideEffect)
        OPTIONAL MATCH (paper:Paper)-[:PROPOSES]->(m)
        OPTIONAL MATCH (paper)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (paper)-[:DISCUSSES]->(c:Concept)
        OPTIONAL MATCH (paper)-[:RELATED_TO]->(rel:Paper)
        OPTIONAL MATCH (paper)-[:DISCUSSES]->(d:Drug)
        OPTIONAL MATCH (doc:Document)-[:EXTRACTS_TO]->(m)
        RETURN m.name AS method, p.name AS protein,
               collect(DISTINCT l.name)[0..10] AS limitations,
               collect(DISTINCT se.name)[0..8] AS side_effects,
               collect(DISTINCT c.name)[0..8] AS concepts,
               collect(DISTINCT a.name)[0..12] AS authors,
               collect(DISTINCT d.name)[0..8] AS related_drugs,
               collect(DISTINCT doc.title)[0..4] AS documents,
               collect(DISTINCT {
                 pmid: paper.pmid, title: paper.title,
                 url: paper.url, year: paper.year
               })[0..10] AS papers,
               collect(DISTINCT {
                 pmid: rel.pmid, title: rel.title,
                 url: rel.url, year: rel.year
               })[0..8] AS related_papers
        ORDER BY m.name
        LIMIT 40
    """,
}


def _normalize_paper(p: dict[str, Any]) -> dict[str, Any] | None:
    pmid = p.get("pmid")
    if not pmid:
        return None
    if not p.get("title") and not p.get("url"):
        return None
    key = str(pmid)
    return {
        "pmid": key,
        "title": p.get("title"),
        "url": p.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{key}/",
        "year": p.get("year"),
    }


def _collect_papers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate paper dicts from record fields or flat paper rows."""
    seen: set[str] = set()
    papers: list[dict[str, Any]] = []
    for rec in records:
        # Flat papers_for_entity rows
        if rec.get("pmid") and rec.get("title"):
            key = str(rec["pmid"])
            if key not in seen:
                seen.add(key)
                papers.append(
                    {
                        "pmid": str(rec["pmid"]),
                        "title": rec.get("title"),
                        "url": rec.get("url")
                        or f"https://pubmed.ncbi.nlm.nih.gov/{rec['pmid']}/",
                        "year": rec.get("year"),
                    }
                )
            continue
        for field in ("papers", "related_papers"):
            for p in rec.get(field) or []:
                if not isinstance(p, dict):
                    continue
                norm = _normalize_paper(p)
                if not norm or norm["pmid"] in seen:
                    continue
                seen.add(norm["pmid"])
                papers.append(norm)
    return papers


def _collect_authors(records: list[dict[str, Any]]) -> list[str]:
    """Deduplicate author names from research_discovery records."""
    seen: set[str] = set()
    authors: list[str] = []
    for rec in records:
        for a in rec.get("authors") or []:
            if not a or a in seen:
                continue
            seen.add(str(a))
            authors.append(str(a))
    return authors


def search(
    *,
    drug: str | None = None,
    disease: str | None = None,
    protein: str | None = None,
    side_effect: str | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    """Structured graph search (no LLM). Used by POST /graph/search."""
    if not is_configured():
        raise Neo4jNotConfiguredError(
            "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
        )

    drug = (drug or "").strip() or None
    disease = (disease or "").strip() or None
    protein = (protein or "").strip() or None
    side_effect = (side_effect or "").strip() or None
    intent = (intent or "").strip() or None

    if intent and intent not in _QUERIES:
        raise ValueError(
            f"Unknown intent '{intent}'. Valid: {', '.join(sorted(_QUERIES))}"
        )

    warning: str | None = None
    if intent:
        chosen = intent
        params: dict[str, Any] = {}
        if chosen in ("drug_neighborhood", "shared_targets", "similar_drugs", "side_effects"):
            if not drug:
                warning = f"missing_required_entity: intent '{chosen}' requires drug"
            else:
                params["drug"] = drug
        elif chosen in ("disease_neighborhood", "treats_disease"):
            if not disease:
                warning = f"missing_required_entity: intent '{chosen}' requires disease"
            else:
                params["disease"] = disease
        elif chosen in ("protein_neighborhood", "multi_hop_research"):
            if not protein:
                warning = f"missing_required_entity: intent '{chosen}' requires protein"
            else:
                params["protein"] = protein
                if disease and chosen == "multi_hop_research":
                    params["disease"] = disease
        elif chosen == "research_discovery":
            if not protein:
                warning = (
                    f"missing_required_entity: intent '{chosen}' requires protein "
                    "(or use NL retrieve with a concept)"
                )
            else:
                params["protein"] = protein
                params["concept"] = None
        elif chosen == "drugs_by_side_effect":
            if not side_effect:
                warning = (
                    f"missing_required_entity: intent '{chosen}' requires side_effect"
                )
            else:
                params["side_effect"] = side_effect
        elif chosen == "papers_for_entity":
            ent = drug or protein or disease or side_effect
            if not ent:
                warning = "missing_required_entity: papers_for_entity needs an entity"
            else:
                params["entity"] = ent
        if warning:
            return _empty_retrieval(chosen, params, warning=warning)
    elif side_effect:
        chosen, params = "drugs_by_side_effect", {"side_effect": side_effect}
    elif protein and disease:
        chosen, params = "multi_hop_research", {"protein": protein, "disease": disease}
    elif protein:
        chosen, params = "protein_neighborhood", {"protein": protein}
    elif disease and drug:
        chosen, params = "drug_neighborhood", {"drug": drug}
    elif disease:
        chosen, params = "treats_disease", {"disease": disease}
    elif drug:
        chosen, params = "drug_neighborhood", {"drug": drug}
    else:
        chosen, params = "list_drugs", {}

    records = run_cypher(_QUERIES[chosen], params)
    result = _format_retrieval(chosen, params, records)
    if not result["paths"] and not result.get("papers"):
        result["evidence_status"] = "none"
        result["warning"] = result.get("warning") or "no_matching_paths"
    return result


def _format_retrieval(
    intent: str,
    params: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    warning: str | None = None,
) -> dict[str, Any]:
    paths: list[str] = []
    entities: set[str] = set()
    papers = _collect_papers(records)
    authors = _collect_authors(records)

    for rec in records:
        if intent == "shared_targets":
            path = (
                f"{rec.get('drug')} -BINDS-> {rec.get('protein')} <-BINDS- "
                f"{rec.get('related_drug')} ; diseases={rec.get('diseases')}"
            )
            entities.update(
                filter(None, [rec.get("drug"), rec.get("protein"), rec.get("related_drug")])
            )
            for d in rec.get("diseases") or []:
                entities.add(d)
            paths.append(path)
        elif intent == "similar_drugs":
            path = (
                f"{rec.get('drug')} -SIMILAR_TO({rec.get('score')})-> "
                f"{rec.get('similar_drug')} ; proteins={rec.get('proteins')} "
                f"diseases={rec.get('diseases')}"
            )
            entities.update(filter(None, [rec.get("drug"), rec.get("similar_drug")]))
            paths.append(path)
        elif intent == "side_effects":
            path = (
                f"{rec.get('drug')} side_effects={rec.get('side_effects')} "
                f"proteins={rec.get('proteins')} diseases={rec.get('diseases')}"
            )
            entities.add(rec.get("drug") or "")
            for s in rec.get("side_effects") or []:
                entities.add(s)
            paths.append(path)
        elif intent == "drugs_by_side_effect":
            path = (
                f"{rec.get('drug')} -HAS_SIDE_EFFECT-> {rec.get('side_effect')} "
                f"; proteins={rec.get('proteins')} diseases={rec.get('diseases')}"
            )
            entities.update(filter(None, [rec.get("drug"), rec.get("side_effect")]))
            paths.append(path)
        elif intent == "treats_disease":
            path = (
                f"{rec.get('drug')} -TREATS-> {rec.get('disease')} "
                f"; proteins={rec.get('proteins')} side_effects={rec.get('side_effects')}"
            )
            entities.update(filter(None, [rec.get("drug"), rec.get("disease")]))
            for p in rec.get("proteins") or []:
                if p:
                    entities.add(p)
                    paths.append(f"{rec.get('drug')} -BINDS-> {p}")
            paths.append(path)
        elif intent == "list_drugs":
            path = (
                f"{rec.get('drug')} treats={rec.get('diseases')} "
                f"targets={rec.get('proteins')}"
            )
            entities.add(rec.get("drug") or "")
            for d in rec.get("diseases") or []:
                entities.add(d)
            paths.append(path)
        elif intent == "disease_neighborhood":
            path = (
                f"{rec.get('disease')} treating_drugs={rec.get('treating_drugs')} "
                f"proteins={rec.get('proteins')} binder_drugs={rec.get('binder_drugs')}"
            )
            entities.add(rec.get("disease") or "")
            for d in (rec.get("treating_drugs") or []) + (rec.get("binder_drugs") or []):
                if d:
                    entities.add(d)
            for p in rec.get("proteins") or []:
                if p:
                    entities.add(p)
            paths.append(path)
        elif intent == "protein_neighborhood":
            path = (
                f"{rec.get('drug')} -BINDS-> {rec.get('protein')} "
                f"; associated={rec.get('associated_diseases')} "
                f"treats={rec.get('treated_diseases')}"
            )
            entities.update(filter(None, [rec.get("drug"), rec.get("protein")]))
            for d in (rec.get("associated_diseases") or []) + (
                rec.get("treated_diseases") or []
            ):
                if d:
                    entities.add(d)
            paths.append(path)
        elif intent == "multi_hop_research":
            drug_name = rec.get("drug")
            protein_name = rec.get("protein")
            entities.update(filter(None, [drug_name, protein_name]))
            if drug_name and protein_name:
                paths.append(f"{drug_name} -BINDS-> {protein_name}")
            for d in rec.get("treated_diseases") or []:
                if d:
                    entities.add(d)
                    paths.append(f"{drug_name} -TREATS-> {d}")
            for d in rec.get("associated_diseases") or []:
                if d and protein_name:
                    entities.add(d)
                    paths.append(f"{protein_name} -ASSOCIATED_WITH-> {d}")
            for s in rec.get("side_effects") or []:
                if s:
                    entities.add(s)
                    paths.append(f"{drug_name} -HAS_SIDE_EFFECT-> {s}")
            for o in rec.get("similar_drugs") or []:
                if o:
                    entities.add(o)
                    paths.append(f"{drug_name} -SIMILAR_TO-> {o}")
        elif intent == "papers_for_entity":
            title = rec.get("title")
            pmid = rec.get("pmid")
            ent = rec.get("entity_name")
            if title and pmid:
                paths.append(
                    f"Paper:{pmid} -DISCUSSES-> {ent} "
                    f"; title={title}"
                )
            if ent:
                entities.add(ent)
        elif intent == "research_discovery":
            method = rec.get("method")
            protein_name = rec.get("protein")
            entities.update(filter(None, [method, protein_name]))
            if method and protein_name:
                paths.append(f"{method} -TARGETS-> {protein_name}")
            for concept_name in rec.get("concepts") or []:
                if concept_name:
                    entities.add(concept_name)
                    for p in rec.get("papers") or []:
                        if isinstance(p, dict) and p.get("pmid"):
                            paths.append(
                                f"Paper:{p['pmid']} -DISCUSSES-> {concept_name}"
                            )
                            break
            for lim in rec.get("limitations") or []:
                if not lim:
                    continue
                entities.add(lim)
                paper_pmids = [
                    str(p["pmid"])
                    for p in (rec.get("papers") or [])
                    if isinstance(p, dict) and p.get("pmid")
                ]
                if paper_pmids and method:
                    paths.append(
                        f"Paper:{paper_pmids[0]} -PROPOSES-> {method} "
                        f"-HAS_LIMITATION-> {lim}"
                    )
                elif method:
                    paths.append(f"{method} -HAS_LIMITATION-> {lim}")
            for se in rec.get("side_effects") or []:
                if se:
                    entities.add(se)
                    for lim in rec.get("limitations") or []:
                        if lim:
                            paths.append(f"{lim} -MANIFESTS_AS-> {se}")
                            break
            for a in rec.get("authors") or []:
                if a:
                    entities.add(a)
                    for p in rec.get("papers") or []:
                        if isinstance(p, dict) and p.get("pmid"):
                            paths.append(f"Paper:{p['pmid']} -AUTHORED_BY-> {a}")
                            break
            for p in rec.get("papers") or []:
                if isinstance(p, dict) and p.get("pmid") and method:
                    paths.append(f"Paper:{p['pmid']} -PROPOSES-> {method}")
            for rp in rec.get("related_papers") or []:
                if not isinstance(rp, dict) or not rp.get("pmid"):
                    continue
                for p in rec.get("papers") or []:
                    if isinstance(p, dict) and p.get("pmid"):
                        paths.append(
                            f"Paper:{p['pmid']} -RELATED_TO-> Paper:{rp['pmid']}"
                        )
                        break
            for drug_name in rec.get("related_drugs") or []:
                if drug_name:
                    entities.add(drug_name)
                    for p in rec.get("papers") or []:
                        if isinstance(p, dict) and p.get("pmid"):
                            paths.append(
                                f"Paper:{p['pmid']} -DISCUSSES-> {drug_name}"
                            )
                            break
            for doc_title in rec.get("documents") or []:
                if doc_title and method:
                    paths.append(f"Document:{doc_title} -EXTRACTS_TO-> {method}")
        elif intent == "drug_neighborhood":
            drug_name = rec.get("drug")
            entities.add(drug_name or "")
            for p in rec.get("proteins") or []:
                if p:
                    entities.add(p)
                    paths.append(f"{drug_name} -BINDS-> {p}")
            for d in rec.get("diseases") or []:
                if d:
                    entities.add(d)
                    paths.append(f"{drug_name} -TREATS-> {d}")
            for d in rec.get("associated_diseases") or []:
                if d:
                    entities.add(d)
            for s in rec.get("side_effects") or []:
                if s:
                    entities.add(s)
                    paths.append(f"{drug_name} -HAS_SIDE_EFFECT-> {s}")
            for o in rec.get("similar_drugs") or []:
                if o:
                    entities.add(o)
                    paths.append(f"{drug_name} -SIMILAR_TO-> {o}")
        else:
            path = (
                f"{rec.get('drug')} -BINDS-> {rec.get('protein')} "
                f"-TREATS/ASSOCIATED-> {rec.get('disease')}"
            )
            entities.update(
                filter(None, [rec.get("drug"), rec.get("protein"), rec.get("disease")])
            )
            paths.append(path)

    # Attach paper DISCUSSES paths for provenance
    for p in papers:
        pmid = p.get("pmid")
        title = p.get("title")
        if pmid and title:
            paths.append(f"Paper:{pmid} title={title} url={p.get('url')}")

    seen: set[str] = set()
    uniq_paths: list[str] = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            uniq_paths.append(p)

    entities.discard("")
    capped = uniq_paths[:MAX_PATHS]
    status = "ok" if capped or papers else "none"
    out: dict[str, Any] = {
        "intent": intent,
        "params": params,
        "records": records,
        "paths": capped,
        "entities": sorted(entities),
        "papers": papers,
        "authors": authors,
        "evidence_status": status,
    }
    if warning:
        out["warning"] = warning
    return out


def _merge_retrievals(*parts: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple retrieval payloads (dedupe paths/entities/papers)."""
    if not parts:
        return _empty_retrieval("merged", {}, warning="no_retrievals")
    base = parts[0]
    paths: list[str] = []
    entities: set[str] = set()
    papers: list[dict[str, Any]] = []
    authors: list[str] = []
    seen_p: set[str] = set()
    seen_pmid: set[str] = set()
    seen_author: set[str] = set()
    records: list[dict[str, Any]] = []
    for part in parts:
        records.extend(part.get("records") or [])
        for p in part.get("paths") or []:
            if p not in seen_p:
                seen_p.add(p)
                paths.append(p)
        entities.update(part.get("entities") or [])
        for paper in part.get("papers") or []:
            pmid = str(paper.get("pmid") or "")
            if pmid and pmid not in seen_pmid:
                seen_pmid.add(pmid)
                papers.append(paper)
        for a in part.get("authors") or []:
            if a and a not in seen_author:
                seen_author.add(a)
                authors.append(a)
    intents = "+".join(sorted({p.get("intent", "") for p in parts if p.get("intent")}))
    params: dict[str, Any] = {}
    for part in parts:
        params.update(part.get("params") or {})
    return {
        "intent": intents or base.get("intent"),
        "params": params,
        "records": records,
        "paths": paths[:MAX_PATHS],
        "entities": sorted(entities),
        "papers": papers,
        "authors": authors,
        "evidence_status": "ok" if paths or papers else "none",
        "warning": next((p.get("warning") for p in parts if p.get("warning")), None),
    }


def retrieve(query: str) -> dict[str, Any]:
    """Run intent-routed Cypher and return structured retrieval payload."""
    if not is_configured():
        raise Neo4jNotConfiguredError(
            "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
        )

    intent, params, warning = _route_intent(query)
    if intent == "research_discovery":
        if warning and "protein" not in params and "concept" not in params:
            return _empty_retrieval(intent, params, warning=warning)
    elif warning and _REQUIRED_PARAM.get(intent) and _REQUIRED_PARAM[intent] not in params:
        return _empty_retrieval(intent, params, warning=warning)

    # Multi-template merge: protein + disease in free text (beyond primary intent)
    q = query.lower()
    _refresh_entity_cache()
    disease = _canonical_name(_extract_entity(q, _ENTITY_CACHE["diseases"]), "Disease")
    protein = _canonical_name(_extract_entity(q, _ENTITY_CACHE["proteins"]), "Protein")

    if intent == "research_discovery":
        cypher_params = {
            "protein": params.get("protein"),
            "concept": params.get("concept"),
        }
        records = run_cypher(_QUERIES["research_discovery"], cypher_params)
        return _format_retrieval(intent, params, records, warning=warning)

    if intent == "multi_hop_research" and params.get("protein"):
        records = run_cypher(_QUERIES["multi_hop_research"], params)
        result = _format_retrieval(intent, params, records, warning=warning)
        # Optionally merge treats_disease if a disease was also named
        if disease and disease != params.get("disease"):
            extra = run_cypher(_QUERIES["treats_disease"], {"disease": disease})
            extra_fmt = _format_retrieval("treats_disease", {"disease": disease}, extra)
            result = _merge_retrievals(result, extra_fmt)
            result["intent"] = "multi_hop_research"
        return result

    if protein and disease and intent in ("protein_neighborhood", "treats_disease"):
        r1 = _format_retrieval(
            "protein_neighborhood",
            {"protein": protein},
            run_cypher(_QUERIES["protein_neighborhood"], {"protein": protein}),
        )
        r2 = _format_retrieval(
            "treats_disease",
            {"disease": disease},
            run_cypher(_QUERIES["treats_disease"], {"disease": disease}),
        )
        merged = _merge_retrievals(r1, r2)
        merged["intent"] = "multi_hop_research"
        return merged

    cypher = _QUERIES[intent]
    records = run_cypher(cypher, params)
    return _format_retrieval(intent, params, records, warning=warning)


def _bounded_context(paths: list[str]) -> str:
    lines: list[str] = []
    total = 0
    for i, p in enumerate(paths, start=1):
        line = f"[{i}] {p}"
        if total + len(line) + 1 > MAX_EVIDENCE_CHARS:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) if lines else "(no graph paths found)"


def _no_evidence_message(retrieval: dict[str, Any]) -> str:
    warning = retrieval.get("warning") or ""
    if warning.startswith("missing_required_entity") or warning.startswith(
        "unknown_entity"
    ):
        return (
            f"No graph evidence could be retrieved. {warning}. "
            "Ask about a seeded entity (e.g. EGFR, Erlotinib, Lung cancer) "
            "or re-run scripts/seed_neo4j.py."
        )
    return (
        "No matching paths were found in the Neo4j knowledge graph for this question. "
        "The graph is a demo-scale seeded dataset — try a known drug (Erlotinib), "
        "protein (EGFR), or disease (Lung cancer). "
        "CURA will not invent biomedical relationships when evidence is missing."
    )


def _synthesize_with_groq(
    question: str, retrieval: dict[str, Any]
) -> tuple[str, str]:
    """Return (answer_text, evidence_status)."""
    paths = retrieval.get("paths") or []
    if not paths and not retrieval.get("papers"):
        return _no_evidence_message(retrieval), "none"

    api_key = os.getenv("GROQ_API_KEY")
    context = _bounded_context(paths)
    entities = ", ".join(retrieval.get("entities") or []) or "(none)"
    papers = retrieval.get("papers") or []
    paper_lines = "\n".join(
        f"- PMID {p.get('pmid')}: {p.get('title')} ({p.get('url')})"
        for p in papers[:8]
        if p.get("pmid")
    ) or "(none)"

    system = (
        "You are CURA GraphRAG, a biomedical knowledge-graph assistant. "
        "Answer ONLY using the numbered Neo4j graph evidence paths below. "
        "Cite path numbers like [1] and entity names from the graph. "
        "Treat paths as retrieved demo-graph facts, not clinically validated claims. "
        "If you add interpretation beyond the paths, label it clearly as interpretation. "
        "If evidence is insufficient, say so. Do not invent drugs, targets, diseases, "
        "side effects, or papers. Be concise (4–8 sentences)."
    )
    user = (
        f"Question: {question}\n\n"
        f"Retrieval intent: {retrieval.get('intent')} params={retrieval.get('params')}\n\n"
        f"Entities: {entities}\n\n"
        f"Curated papers (DISCUSSES links only — not SUPPORTS/CONTRADICTS):\n{paper_lines}\n\n"
        f"Graph evidence paths:\n{context}\n"
    )

    if not api_key:
        bullets = "\n".join(f"• {p}" for p in paths[:12])
        return (
            f"Graph evidence (intent={retrieval.get('intent')}):\n{bullets}\n\n"
            "(Set GROQ_API_KEY to enable natural-language synthesis. "
            "These paths are seeded demo-graph facts, not clinical validation.)"
        ), "paths_only"

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq
        from pydantic import SecretStr

        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key=SecretStr(api_key),
            temperature=0.0,
        )
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        text = getattr(msg, "content", str(msg))
        if not paths and text:
            return _no_evidence_message(retrieval), "none"
        return text, "grounded"
    except Exception as exc:
        logger.warning("[GraphRAG] Groq synthesis failed: %s", exc)
        bullets = "\n".join(f"• {p}" for p in paths[:12])
        return (
            f"LLM synthesis unavailable ({exc}). Raw graph evidence:\n{bullets}"
        ), "error"


def answer(query: str) -> dict[str, Any]:
    """Full GraphRAG: retrieve from Neo4j then synthesize an answer."""
    q = (query or "").strip()
    if not q:
        raise ValueError("query must be a non-empty string")

    retrieval = retrieve(q)
    text, status = _synthesize_with_groq(q, retrieval)
    # Post-check: never present LLM prose as grounded when paths are empty
    if not retrieval.get("paths") and not retrieval.get("papers"):
        text = _no_evidence_message(retrieval)
        status = "none"

    out: dict[str, Any] = {
        "answer": text,
        "paths": retrieval.get("paths") or [],
        "entities": retrieval.get("entities") or [],
        "intent": retrieval.get("intent"),
        "params": retrieval.get("params") or {},
        "papers": retrieval.get("papers") or [],
        "authors": retrieval.get("authors") or [],
        "evidence_status": status,
    }
    if retrieval.get("warning"):
        out["warning"] = retrieval["warning"]
    return out
