"""Drug–protein–disease knowledge graph for CURA network visualization.

Primary source: Neo4j Aura (when NEO4J_* is configured).
Fallback: local CSV so the API still boots without Aura for non-demo work.
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

from models.graph.neo4j_client import Neo4jNotConfiguredError, is_configured, run_cypher

logger = logging.getLogger(__name__)

_CSV_PATH = Path("data/drug_protein_disease.csv")

_NODE_COLORS = {
    "drug": "#06b6d4",
    "protein": "#3b82f6",
    "disease": "#10b981",
    "sideeffect": "#f59e0b",
}


class KnowledgeGraphEngine:
    def __init__(self) -> None:
        self.data = pd.DataFrame()
        self.graph = nx.Graph()
        self._use_neo4j = is_configured()
        if not self._use_neo4j:
            if _CSV_PATH.exists():
                self.data = pd.read_csv(_CSV_PATH)
                logger.info(
                    "[KG] Neo4j not configured — using CSV fallback (%d rows).",
                    len(self.data),
                )
            else:
                logger.warning("[KG] No Neo4j and no CSV at %s.", _CSV_PATH)
        else:
            logger.info("[KG] Using Neo4j as knowledge-graph source.")

    def _ensure_csv_loaded(self) -> None:
        if self.data.empty and _CSV_PATH.exists():
            self.data = pd.read_csv(_CSV_PATH)
            logger.info("[KG] Loaded CSV fallback (%d rows).", len(self.data))

    def build_graph(self, drug: str | None = None, disease: str | None = None):
        self.graph = nx.Graph()
        if self._use_neo4j:
            try:
                return self._build_from_neo4j(drug, disease)
            except Neo4jNotConfiguredError:
                logger.warning("[KG] Neo4j became unavailable — falling back to CSV.")
                self._ensure_csv_loaded()
                return self._build_from_csv(drug, disease)
            except Exception as exc:
                # DNS / Aura pause / transient network: keep Network usable via CSV.
                # GraphRAG still requires a live Neo4j connection separately.
                logger.warning(
                    "[KG] Neo4j query failed (%s) — falling back to CSV for visualization.",
                    exc,
                )
                try:
                    from models.graph.neo4j_client import close_driver

                    close_driver()  # drop stale pool so the next request can reconnect
                except Exception:
                    pass
                self._ensure_csv_loaded()
                graph = self._build_from_csv(drug, disease)
                if graph is not None:
                    return graph
                # No CSV match either — surface a clear error
                raise RuntimeError(
                    f"Neo4j unreachable ({exc}) and no CSV neighborhood matched "
                    f"drug={drug!r} disease={disease!r}. Check Aura is running and "
                    f"DNS can resolve NEO4J_URI, or use a seeded drug/disease name."
                ) from exc
        return self._build_from_csv(drug, disease)

    def _build_from_neo4j(self, drug: str | None, disease: str | None):
        drug = drug.strip() if drug and drug.strip() else None
        disease = disease.strip() if disease and disease.strip() else None
        records: list[dict] = []

        if drug and disease:
            records = run_cypher(
                """
                MATCH (d:Drug)-[:TREATS]->(dis:Disease)
                WHERE toLower(d.name) = toLower($drug)
                  AND toLower(dis.name) = toLower($disease)
                OPTIONAL MATCH (d)-[:BINDS]->(p:Protein)
                OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(adis:Disease)
                OPTIONAL MATCH (d)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
                OPTIONAL MATCH (d)-[sim:SIMILAR_TO]->(other:Drug)
                RETURN d.name AS drug,
                       collect(DISTINCT p.name) AS proteins,
                       collect(DISTINCT dis.name) + collect(DISTINCT adis.name) AS diseases,
                       collect(DISTINCT s.name) AS side_effects,
                       collect(DISTINCT {name: other.name, score: sim.score}) AS similar
                LIMIT 20
                """,
                {"drug": drug, "disease": disease},
            )
            if not records:
                records = []

        if not (drug and disease and records):
            if drug:
                records = run_cypher(
                    """
                    MATCH (d:Drug)
                    WHERE toLower(d.name) = toLower($drug)
                    OPTIONAL MATCH (d)-[:BINDS]->(p:Protein)
                    OPTIONAL MATCH (d)-[:TREATS]->(dis:Disease)
                    OPTIONAL MATCH (p)-[:ASSOCIATED_WITH]->(adis:Disease)
                    OPTIONAL MATCH (d)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
                    OPTIONAL MATCH (d)-[sim:SIMILAR_TO]->(other:Drug)
                    RETURN d.name AS drug,
                           collect(DISTINCT p.name) AS proteins,
                           collect(DISTINCT dis.name) + collect(DISTINCT adis.name) AS diseases,
                           collect(DISTINCT s.name) AS side_effects,
                           collect(DISTINCT {name: other.name, score: sim.score}) AS similar
                    LIMIT 20
                    """,
                    {"drug": drug},
                )
            elif disease:
                records = run_cypher(
                    """
                    MATCH (dis:Disease)
                    WHERE toLower(dis.name) = toLower($disease)
                    OPTIONAL MATCH (d:Drug)-[:TREATS]->(dis)
                    OPTIONAL MATCH (d)-[:BINDS]->(p:Protein)
                    OPTIONAL MATCH (p2:Protein)-[:ASSOCIATED_WITH]->(dis)
                    OPTIONAL MATCH (d2:Drug)-[:BINDS]->(p2)
                    OPTIONAL MATCH (d)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
                    OPTIONAL MATCH (d)-[sim:SIMILAR_TO]->(other:Drug)
                    RETURN coalesce(d.name, d2.name) AS drug,
                           collect(DISTINCT coalesce(p.name, p2.name)) AS proteins,
                           collect(DISTINCT dis.name) AS diseases,
                           collect(DISTINCT s.name) AS side_effects,
                           collect(DISTINCT {name: other.name, score: sim.score}) AS similar
                    LIMIT 40
                    """,
                    {"disease": disease},
                )
            else:
                records = run_cypher(
                    """
                    MATCH (d:Drug)-[:TREATS]->(dis:Disease)
                    OPTIONAL MATCH (d)-[:BINDS]->(p:Protein)
                    RETURN d.name AS drug,
                           collect(DISTINCT p.name) AS proteins,
                           collect(DISTINCT dis.name) AS diseases,
                           [] AS side_effects, [] AS similar
                    LIMIT 40
                    """
                )

        if not records:
            logger.info("[KG] No Neo4j matches for drug=%s disease=%s", drug, disease)
            return None

        for rec in records:
            d_node = rec.get("drug")
            if not d_node:
                continue
            proteins = [p for p in (rec.get("proteins") or []) if p]
            diseases = [d for d in (rec.get("diseases") or []) if d]
            # Deduplicate diseases while preserving order
            seen_dis: set[str] = set()
            uniq_diseases: list[str] = []
            for d in diseases:
                if d not in seen_dis:
                    seen_dis.add(d)
                    uniq_diseases.append(d)

            self.graph.add_node(d_node, type="drug", label=d_node)
            for p_node in proteins:
                self.graph.add_node(p_node, type="protein", label=p_node)
                self.graph.add_edge(d_node, p_node, label="binds")
            for dis_node in uniq_diseases:
                self.graph.add_node(dis_node, type="disease", label=dis_node)
                self.graph.add_edge(d_node, dis_node, label="treats")
                for p_node in proteins:
                    self.graph.add_edge(p_node, dis_node, label="associated")

            for se in rec.get("side_effects") or []:
                if not se:
                    continue
                self.graph.add_node(se, type="sideeffect", label=se)
                self.graph.add_edge(d_node, se, label="side_effect")

            for sim in rec.get("similar") or []:
                if not isinstance(sim, dict):
                    continue
                other = sim.get("name")
                if not other:
                    continue
                score = sim.get("score")
                label = f"similar:{score:.2f}" if isinstance(score, (int, float)) else "similar"
                self.graph.add_node(other, type="drug", label=other)
                self.graph.add_edge(d_node, other, label=label)

        return self.graph

    def _build_from_csv(self, drug: str | None, disease: str | None):
        if self.data.empty:
            return None

        if drug and disease:
            subset = self.data[
                (self.data["drug"].str.lower() == drug.lower())
                & (self.data["disease"].str.lower() == disease.lower())
            ]
            if not subset.empty:
                self._add_csv_subset(subset)
                return self.graph

        if drug:
            subset = self.data[self.data["drug"].str.lower() == drug.lower()]
            if not subset.empty:
                self._add_csv_subset(subset)
                return self.graph

        if disease:
            subset = self.data[self.data["disease"].str.lower() == disease.lower()]
            if not subset.empty:
                self._add_csv_subset(subset)
                return self.graph

        logger.info("No interactions found for Drug: %s, Disease: %s", drug, disease)
        return None

    def _add_csv_subset(self, subset: pd.DataFrame) -> None:
        if len(subset) > 50:
            subset = subset.head(50)
        for _, row in subset.iterrows():
            d_node = row["drug"]
            p_node = row["protein"]
            dis_node = row["disease"]
            self.graph.add_node(d_node, type="drug", label=d_node)
            self.graph.add_node(p_node, type="protein", label=p_node)
            self.graph.add_node(dis_node, type="disease", label=dis_node)
            self.graph.add_edge(d_node, p_node, label="binds")
            self.graph.add_edge(d_node, dis_node, label="treats")
            self.graph.add_edge(p_node, dis_node, label="associated")

    def visualize(self, drug: str | None = None, disease: str | None = None) -> None:
        graph = self.build_graph(drug, disease)

        net = Network(height="600px", width="100%", bgcolor="#0f172a", font_color="white")
        if graph is None:
            net.add_node("Error", label="No target pathways found for this query", color="gray")
            net.write_html("drug_knowledge_graph.html")
            return

        net.force_atlas_2based()
        for node, data in graph.nodes(data=True):
            ntype = data.get("type", "disease")
            color = _NODE_COLORS.get(ntype, "#10b981")
            size = 30 if ntype == "drug" else 20 if ntype == "protein" else 22
            net.add_node(node, label=node, color=color, size=size, shadow=True)

        for edge in graph.edges(data=True):
            net.add_edge(edge[0], edge[1], color="#475569", width=2, label=edge[2].get("label"))

        net.write_html("drug_knowledge_graph.html")
        logger.info("Graph saved as drug_knowledge_graph.html")
