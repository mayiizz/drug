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

# Cache of entity names loaded from Neo4j (refreshed periodically).
_ENTITY_CACHE: dict[str, set[str]] = {
    "drugs": set(),
    "diseases": set(),
    "proteins": set(),
    "side_effects": set(),
}
_ENTITY_CACHE_TS: float = 0.0
_ENTITY_CACHE_TTL = 300.0  # seconds


def _refresh_entity_cache(force: bool = False) -> None:
    """Load Drug/Disease/Protein/SideEffect names from Neo4j into memory."""
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
        rows = run_cypher(
            """
            MATCH (n)
            WHERE n:Drug OR n:Disease OR n:Protein OR n:SideEffect
            RETURN labels(n)[0] AS label, collect(DISTINCT toLower(n.name)) AS names
            """
        )
        mapping = {
            "Drug": "drugs",
            "Disease": "diseases",
            "Protein": "proteins",
            "SideEffect": "side_effects",
        }
        fresh = {k: set() for k in _ENTITY_CACHE}
        for row in rows:
            key = mapping.get(row.get("label") or "")
            if not key:
                continue
            fresh[key] = {str(n) for n in (row.get("names") or []) if n}
        _ENTITY_CACHE.update(fresh)
        _ENTITY_CACHE_TS = now
        logger.info(
            "[GraphRAG] Entity cache: drugs=%d diseases=%d proteins=%d side_effects=%d",
            len(fresh["drugs"]),
            len(fresh["diseases"]),
            len(fresh["proteins"]),
            len(fresh["side_effects"]),
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
    # Sensible display fallback
    if matched_lower.isupper() or len(matched_lower) <= 6:
        return matched_lower.upper()
    return matched_lower.title()


def _route_intent(query: str) -> tuple[str, dict[str, Any]]:
    """Return (intent, params) for Cypher template selection."""
    _refresh_entity_cache()
    q = query.lower()

    drug = _canonical_name(_extract_entity(q, _ENTITY_CACHE["drugs"]), "Drug")
    disease = _canonical_name(_extract_entity(q, _ENTITY_CACHE["diseases"]), "Disease")
    protein = _canonical_name(_extract_entity(q, _ENTITY_CACHE["proteins"]), "Protein")
    side_effect = _canonical_name(
        _extract_entity(q, _ENTITY_CACHE["side_effects"]), "SideEffect"
    )

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
        return "list_drugs", {}

    if any(w in q for w in ("side effect", "adverse", "toxicity", "adr")):
        if side_effect and not drug:
            return "drugs_by_side_effect", {"side_effect": side_effect}
        return "side_effects", {"drug": drug or "Erlotinib"}

    if side_effect and any(w in q for w in ("cause", "causes", "with", "have", "has")):
        return "drugs_by_side_effect", {"side_effect": side_effect}

    if any(w in q for w in ("similar", "analog", "like")):
        return "similar_drugs", {"drug": drug or "Erlotinib"}

    if any(w in q for w in ("share", "common target", "same target", "shared")):
        return "shared_targets", {"drug": drug or "Erlotinib"}

    if any(w in q for w in ("treat", "treats", "treatment", "indication", "therapy")):
        if disease:
            return "treats_disease", {"disease": disease}
        if drug:
            return "drug_neighborhood", {"drug": drug}

    if protein and ("drug" in q or "inhibit" in q or "bind" in q or "target" in q):
        return "protein_neighborhood", {"protein": protein}

    if disease and not drug:
        if any(w in q for w in ("drug", "treat", "therapy", "medicine")):
            return "treats_disease", {"disease": disease}
        return "disease_neighborhood", {"disease": disease}

    if drug:
        return "drug_neighborhood", {"drug": drug}

    if disease:
        return "treats_disease", {"disease": disease}

    if protein:
        return "protein_neighborhood", {"protein": protein}

    if side_effect:
        return "drugs_by_side_effect", {"side_effect": side_effect}

    # Free-text: Capitalized token that looks like a drug name
    m = re.search(r"\b([A-Z][a-z]+(?:ib|inib|mab|tinib|ciclib|statin|olol|pril)?)\b", query)
    if m:
        return "drug_neighborhood", {"drug": m.group(1)}

    if any(w in q for w in ("overview", "summary", "graph", "knowledge")):
        return "overview", {}

    return "list_drugs", {}


_QUERIES: dict[str, str] = {
    "drug_neighborhood": """
        MATCH (d:Drug)
        WHERE toLower(d.name) = toLower($drug)
        OPTIONAL MATCH (d)-[:BINDS]->(p:Protein)
        OPTIONAL MATCH (d)-[:TREATS]->(dis:Disease)
        OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(adis:Disease)
        OPTIONAL MATCH (d)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        OPTIONAL MATCH (d)-[sim:SIMILAR_TO]->(other:Drug)
        RETURN d.name AS drug,
               collect(DISTINCT p.name)[0..12] AS proteins,
               collect(DISTINCT dis.name)[0..12] AS diseases,
               collect(DISTINCT adis.name)[0..8] AS associated_diseases,
               collect(DISTINCT s.name)[0..10] AS side_effects,
               collect(DISTINCT other.name)[0..10] AS similar_drugs
        LIMIT 5
    """,
    "disease_neighborhood": """
        MATCH (dis:Disease)
        WHERE toLower(dis.name) = toLower($disease)
        OPTIONAL MATCH (d:Drug)-[:TREATS]->(dis)
        OPTIONAL MATCH (p:Protein)-[:ASSOCIATED_WITH]->(dis)
        OPTIONAL MATCH (d2:Drug)-[:BINDS]->(p)
        RETURN dis.name AS disease,
               collect(DISTINCT d.name)[0..20] AS treating_drugs,
               collect(DISTINCT p.name)[0..15] AS proteins,
               collect(DISTINCT d2.name)[0..20] AS binder_drugs
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
        RETURN d.name AS drug, p.name AS protein,
               collect(DISTINCT dis.name) AS associated_diseases,
               collect(DISTINCT td.name) AS treated_diseases
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
}


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

    if intent:
        chosen = intent
        params: dict[str, Any] = {}
        if chosen in ("drug_neighborhood", "shared_targets", "similar_drugs", "side_effects"):
            params["drug"] = drug or "Erlotinib"
        elif chosen in ("disease_neighborhood", "treats_disease"):
            params["disease"] = disease or "Lung cancer"
        elif chosen == "protein_neighborhood":
            params["protein"] = protein or "EGFR"
        elif chosen == "drugs_by_side_effect":
            params["side_effect"] = side_effect or "Nausea"
    elif side_effect:
        chosen, params = "drugs_by_side_effect", {"side_effect": side_effect}
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
    return _format_retrieval(chosen, params, records)


def _format_retrieval(
    intent: str, params: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    paths: list[str] = []
    entities: set[str] = set()

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
            for d in (rec.get("associated_diseases") or []) + (rec.get("treated_diseases") or []):
                if d:
                    entities.add(d)
            paths.append(path)
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

    seen: set[str] = set()
    uniq_paths: list[str] = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            uniq_paths.append(p)

    entities.discard("")
    return {
        "intent": intent,
        "params": params,
        "records": records,
        "paths": uniq_paths[:60],
        "entities": sorted(entities),
    }


def retrieve(query: str) -> dict[str, Any]:
    """Run intent-routed Cypher and return structured retrieval payload."""
    if not is_configured():
        raise Neo4jNotConfiguredError(
            "Neo4j is not configured. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
        )

    intent, params = _route_intent(query)
    cypher = _QUERIES[intent]
    records = run_cypher(cypher, params)
    return _format_retrieval(intent, params, records)


def _synthesize_with_groq(question: str, retrieval: dict[str, Any]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    context = "\n".join(f"- {p}" for p in retrieval["paths"]) or "(no graph paths found)"
    entities = ", ".join(retrieval["entities"]) or "(none)"

    system = (
        "You are CURA GraphRAG, a biomedical knowledge-graph assistant. "
        "Answer ONLY using the provided Neo4j graph paths. "
        "Cite entity names from the graph. If the graph lacks evidence, say so. "
        "Be concise (4–8 sentences)."
    )
    user = (
        f"Question: {question}\n\n"
        f"Retrieval intent: {retrieval['intent']} params={retrieval['params']}\n\n"
        f"Entities: {entities}\n\n"
        f"Graph paths:\n{context}\n"
    )

    if not api_key:
        if not retrieval["paths"]:
            return (
                "No matching paths were found in the Neo4j knowledge graph for this question. "
                "Try a seeded drug (e.g. Erlotinib) or disease (e.g. Lung cancer). "
                "Set GROQ_API_KEY for LLM synthesis."
            )
        bullets = "\n".join(f"• {p}" for p in retrieval["paths"][:12])
        return (
            f"Graph evidence (intent={retrieval['intent']}):\n{bullets}\n\n"
            "(Set GROQ_API_KEY to enable natural-language synthesis.)"
        )

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
        return getattr(msg, "content", str(msg))
    except Exception as exc:
        logger.warning("[GraphRAG] Groq synthesis failed: %s", exc)
        bullets = "\n".join(f"• {p}" for p in retrieval["paths"][:12])
        return (
            f"LLM synthesis unavailable ({exc}). Raw graph evidence:\n{bullets}"
        )


def answer(query: str) -> dict[str, Any]:
    """Full GraphRAG: retrieve from Neo4j then synthesize an answer."""
    q = (query or "").strip()
    if not q:
        raise ValueError("query must be a non-empty string")

    retrieval = retrieve(q)
    text = _synthesize_with_groq(q, retrieval)
    return {
        "answer": text,
        "paths": retrieval["paths"],
        "entities": retrieval["entities"],
        "intent": retrieval["intent"],
        "params": retrieval["params"],
    }
