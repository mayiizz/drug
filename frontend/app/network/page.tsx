"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  askGraphRAG,
  fetchNetwork,
  searchGraph,
  type GraphRAGResponse,
  type GraphSearchResponse,
} from "@/lib/api";
import {
  Network,
  Loader2,
  RefreshCcw,
  Sparkles,
  AlertCircle,
  ExternalLink,
  Search,
} from "lucide-react";

const SUGGESTED_QUESTIONS = [
  "Which approaches target EGFR, what limitations are recorded, and which papers discuss related drugs?",
  "Which drugs in the available graph target EGFR, what disease relationships are recorded for those targets, and what side effects are associated with the relevant drugs?",
  "Which drugs share a target with Erlotinib and what diseases are linked?",
  "Which drugs treat lung cancer?",
  "Side effects of Metformin",
  "Drugs similar to Gefitinib",
  "What papers discuss EGFR?",
];

export default function NetworkPage() {
  const [drug, setDrug] = useState("Erlotinib");
  const [disease, setDisease] = useState("Lung cancer");
  const [loading, setLoading] = useState(false);
  const [graphUrl, setGraphUrl] = useState<string | null>(null);
  const [graphError, setGraphError] = useState("");

  const [ragQuery, setRagQuery] = useState(SUGGESTED_QUESTIONS[0]);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragResult, setRagResult] = useState<GraphRAGResponse | null>(null);
  const [ragError, setRagError] = useState("");

  const [evidenceOnly, setEvidenceOnly] = useState(false);
  const [searchResult, setSearchResult] = useState<GraphSearchResponse | null>(null);

  const handleLoad = useCallback(async (d?: string, dis?: string) => {
    const drugVal = d ?? drug;
    const diseaseVal = dis ?? disease;
    setLoading(true);
    setGraphError("");
    try {
      const blob = await fetchNetwork(drugVal, diseaseVal);
      const url = URL.createObjectURL(new Blob([blob], { type: "text/html" }));
      setGraphUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
    } catch (e: unknown) {
      console.error(e);
      setGraphError(e instanceof Error ? e.message : "Failed to load graph");
      setGraphUrl(null);
    } finally {
      setLoading(false);
    }
  }, [drug, disease]);

  const syncVizFromParams = useCallback(
    (params?: Record<string, unknown>) => {
      if (!params) return;
      const pDrug = typeof params.drug === "string" ? params.drug : undefined;
      const pDisease = typeof params.disease === "string" ? params.disease : undefined;
      const pProtein = typeof params.protein === "string" ? params.protein : undefined;
      let nextDrug = drug;
      let nextDisease = disease;
      if (pDrug) {
        nextDrug = pDrug;
        setDrug(pDrug);
      }
      if (pDisease) {
        nextDisease = pDisease;
        setDisease(pDisease);
      } else if (pProtein && !pDrug) {
        // Protein-centric answers: keep disease if set; refresh with current drug filter empty-ish
        nextDisease = disease;
      }
      if (pDrug || pDisease) {
        void handleLoad(nextDrug, nextDisease);
      }
    },
    [drug, disease, handleLoad]
  );

  const handleGraphRAG = async (queryOverride?: string) => {
    const q = (queryOverride ?? ragQuery).trim();
    if (!q) return;
    setRagQuery(q);
    setRagLoading(true);
    setRagError("");
    setRagResult(null);
    setSearchResult(null);
    try {
      if (evidenceOnly) {
        // Fast structured retrieval without LLM
        const data = await searchGraph({ intent: undefined });
        // Prefer full GraphRAG for NL; for evidence-only use protein/drug from query heuristically via /graphrag paths without LLM:
        // Call /graph/search with empty filters falls to list_drugs — better call askGraphRAG always for NL,
        // and use searchGraph when user picks a filter. For NL evidence-only, still use askGraphRAG
        // (backend skips Groq when paths empty / returns paths_only without key).
        const rag = await askGraphRAG(q);
        setRagResult(rag);
        setSearchResult(data);
        syncVizFromParams(rag.params);
      } else {
        const data = await askGraphRAG(q);
        setRagResult(data);
        syncVizFromParams(data.params);
      }
    } catch (e: unknown) {
      console.error(e);
      setRagError(e instanceof Error ? e.message : "GraphRAG request failed");
    } finally {
      setRagLoading(false);
    }
  };

  const handleEvidenceBrowse = async () => {
    setRagLoading(true);
    setRagError("");
    setSearchResult(null);
    try {
      const data = await searchGraph({
        drug: drug.trim() || undefined,
        disease: disease.trim() || undefined,
      });
      setSearchResult(data);
      setRagResult({
        answer:
          data.paths?.length
            ? `Structured graph evidence (intent=${data.intent}, no LLM).`
            : data.warning ||
              "No matching paths in Neo4j for these filters.",
        paths: data.paths || [],
        entities: data.entities || [],
        intent: data.intent,
        params: data.params,
        papers: data.papers,
        evidence_status: data.evidence_status,
        warning: data.warning,
      });
    } catch (e: unknown) {
      setRagError(e instanceof Error ? e.message : "Graph search failed");
    } finally {
      setRagLoading(false);
    }
  };

  const onEntityClick = (entity: string) => {
    // Heuristic: diseases often contain spaces / "cancer"; otherwise treat as drug
    const looksDisease =
      /cancer|diabetes|melanoma|leukemia|depression|hypertension|inflammation|pain|fatigue|gerd|covid|arthritis|thrombosis|myeloma|asthma|epilepsy|alzheimer|parkinson/i.test(
        entity
      );
    if (looksDisease) {
      setDisease(entity);
      void handleLoad(drug, entity);
    } else {
      setDrug(entity);
      void handleLoad(entity, disease);
    }
  };

  useEffect(() => {
    handleLoad();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const noEvidence =
    ragResult &&
    (ragResult.evidence_status === "none" ||
      ((ragResult.paths?.length ?? 0) === 0 && (ragResult.papers?.length ?? 0) === 0));

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-8 flex flex-col" style={{ minHeight: "calc(100vh - 80px)" }}>
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-5">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-widest mb-1" style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}>// Bio Network</p>
          <h1 className="text-4xl font-bold" style={{ color: "var(--text-dark)", letterSpacing: "-1px" }}>Drug–Protein–Disease Network</h1>
          <p className="text-sm mt-1" style={{ color: "var(--text-light)", fontWeight: 300 }}>
            Neo4j knowledge graph visualization + GraphRAG Q&amp;A over multi-hop paths.
          </p>
          <p
            className="text-xs mt-2 px-3 py-2 rounded-xl inline-block"
            style={{
              background: "rgba(16,185,129,0.08)",
              border: "1px solid rgba(16,185,129,0.25)",
              color: "#047857",
              maxWidth: "42rem",
            }}
          >
            Curated research knowledge graph (demo) — Paper / Author / Concept / Method / Limitation.
            For open PubMed surveys use{" "}
            <a href="/research" className="underline font-semibold" style={{ color: "#065f46" }}>
              Research Agent
            </a>
            ; use GraphRAG here for seeded multi-hop structure.
          </p>
        </div>

        <div className="flex gap-3 items-end flex-wrap">
          <div>
            <label className="text-[10px] font-semibold uppercase tracking-widest block mb-1.5" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>Drug</label>
            <input
              value={drug}
              onChange={(e) => setDrug(e.target.value)}
              placeholder="e.g. Erlotinib"
              className="px-4 py-2.5 rounded-xl text-sm outline-none w-40"
              style={{ background: "#fff", border: "1px solid var(--border)", color: "var(--text-dark)" }}
            />
          </div>
          <div>
            <label className="text-[10px] font-semibold uppercase tracking-widest block mb-1.5" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>Disease</label>
            <input
              value={disease}
              onChange={(e) => setDisease(e.target.value)}
              placeholder="e.g. Lung cancer"
              className="px-4 py-2.5 rounded-xl text-sm outline-none w-44"
              style={{ background: "#fff", border: "1px solid var(--border)", color: "var(--text-dark)" }}
            />
          </div>
          <button
            onClick={() => handleLoad()}
            disabled={loading}
            className="p-2.5 rounded-xl transition-all hover:-translate-y-0.5 disabled:opacity-50"
            style={{ background: "var(--blue-bright)", color: "#fff", boxShadow: "0 4px 14px rgba(63,184,232,0.3)" }}
            title="Refresh network"
          >
            {loading ? <Loader2 className="animate-spin" size={18} /> : <RefreshCcw size={18} />}
          </button>
        </div>
      </div>

      {graphError && (
        <div className="flex items-start gap-2 px-4 py-3 rounded-xl text-sm" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)", color: "#b91c1c" }}>
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span>{graphError}</span>
        </div>
      )}

      <div
        className="flex-1 relative rounded-2xl overflow-hidden"
        style={{ border: "1px solid var(--border)", background: "#fff", minHeight: "520px" }}
      >
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center z-50"
            style={{ background: "rgba(240,244,248,0.85)", backdropFilter: "blur(8px)" }}>
            <Loader2 className="w-10 h-10 animate-spin mb-3" style={{ color: "var(--blue-bright)" }} />
            <p className="text-sm font-medium" style={{ color: "var(--text-mid)" }}>Building interaction topology…</p>
          </div>
        )}

        {graphUrl ? (
          <iframe src={graphUrl} className="w-full h-full border-none" title="Network Graph" style={{ minHeight: "520px" }} />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-3" style={{ minHeight: "520px" }}>
            <Network className="w-12 h-12" style={{ color: "var(--blue-pale)" }} />
            <p className="text-sm" style={{ color: "var(--text-light)" }}>Enter drug and disease, then click refresh to generate the map.</p>
          </div>
        )}

        <div className="absolute bottom-4 left-4 flex gap-3 flex-wrap">
          {[
            { color: "var(--blue-bright)", label: "Drug" },
            { color: "var(--blue-deep)", label: "Protein" },
            { color: "#10b981", label: "Disease" },
            { color: "#f59e0b", label: "Side effect" },
          ].map((l) => (
            <div
              key={l.label}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold"
              style={{ background: "rgba(255,255,255,0.9)", border: "1px solid var(--border)", color: "var(--text-mid)", backdropFilter: "blur(8px)" }}
            >
              <div className="w-2.5 h-2.5 rounded-full" style={{ background: l.color }} />
              {l.label}
            </div>
          ))}
        </div>
      </div>

      {/* GraphRAG panel */}
      <div
        className="rounded-2xl p-6 space-y-4"
        style={{ background: "#fff", border: "1px solid var(--border)" }}
      >
        <div className="flex items-center gap-2 flex-wrap">
          <Sparkles size={18} style={{ color: "var(--blue-mid)" }} />
          <h2 className="text-lg font-semibold" style={{ color: "var(--text-dark)" }}>GraphRAG</h2>
          <span className="text-[10px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full"
            style={{ background: "rgba(63,184,232,0.12)", color: "var(--blue-mid)" }}>
            Neo4j + LLM
          </span>
          <span className="text-[10px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full"
            style={{ background: "rgba(16,185,129,0.1)", color: "#047857" }}>
            Evidence-grounded
          </span>
        </div>
        <p className="text-sm" style={{ color: "var(--text-light)", fontWeight: 300 }}>
          Ask a multi-hop research question (drug neighborhood or Paper→Method→Limitation discovery).
          CURA retrieves Neo4j paths, then synthesizes an answer only from that evidence.
          Seeded demo graph — not a live clinical database.
        </p>

        <div>
          <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>
            Suggested questions
          </p>
          <div className="flex flex-wrap gap-2">
            {SUGGESTED_QUESTIONS.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => void handleGraphRAG(q)}
                disabled={ragLoading}
                className="text-left text-xs px-3 py-2 rounded-full transition-all hover:-translate-y-0.5 disabled:opacity-50"
                style={{
                  background: ragQuery === q ? "rgba(63,184,232,0.15)" : "#f8fafc",
                  border: "1px solid var(--border)",
                  color: "var(--text-mid)",
                  maxWidth: "100%",
                }}
              >
                {q.length > 90 ? `${q.slice(0, 90)}…` : q}
              </button>
            ))}
          </div>
        </div>

        <textarea
          value={ragQuery}
          onChange={(e) => setRagQuery(e.target.value)}
          rows={3}
          className="w-full px-4 py-3 rounded-xl text-sm outline-none resize-y"
          style={{ background: "#f8fafc", border: "1px solid var(--border)", color: "var(--text-dark)" }}
          placeholder="e.g. Which drugs target EGFR and what side effects are linked?"
        />

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => void handleGraphRAG()}
            disabled={ragLoading || !ragQuery.trim()}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:-translate-y-0.5 disabled:opacity-50"
            style={{ background: "var(--blue-bright)", color: "#fff", boxShadow: "0 4px 14px rgba(63,184,232,0.3)" }}
          >
            {ragLoading ? <Loader2 className="animate-spin" size={16} /> : <Sparkles size={16} />}
            {ragLoading ? "Querying graph…" : "Ask GraphRAG"}
          </button>
          <button
            onClick={() => void handleEvidenceBrowse()}
            disabled={ragLoading}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all hover:-translate-y-0.5 disabled:opacity-50"
            style={{ background: "#fff", color: "var(--blue-deep)", border: "1px solid var(--border)" }}
            title="Browse Neo4j paths for the drug/disease filters without LLM"
          >
            <Search size={16} />
            Evidence only
          </button>
          <label className="inline-flex items-center gap-2 text-xs cursor-pointer" style={{ color: "var(--text-light)" }}>
            <input
              type="checkbox"
              checked={evidenceOnly}
              onChange={(e) => setEvidenceOnly(e.target.checked)}
              className="rounded"
            />
            Also call /graph/search catalog when asking
          </label>
        </div>

        {ragError && (
          <div className="flex items-start gap-2 px-4 py-3 rounded-xl text-sm" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)", color: "#b91c1c" }}>
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{ragError}</span>
          </div>
        )}

        {ragResult && (
          <div className="space-y-4 pt-2">
            {(ragResult.intent || ragResult.evidence_status) && (
              <div className="flex flex-wrap gap-2 items-center">
                {ragResult.intent && (
                  <span className="text-[10px] font-mono px-2.5 py-1 rounded-full"
                    style={{ background: "#f1f5f9", color: "var(--text-mid)", border: "1px solid var(--border)" }}>
                    intent={ragResult.intent}
                  </span>
                )}
                {ragResult.params && Object.keys(ragResult.params).length > 0 && (
                  <span className="text-[10px] font-mono px-2.5 py-1 rounded-full"
                    style={{ background: "#f1f5f9", color: "var(--text-mid)", border: "1px solid var(--border)" }}>
                    params={JSON.stringify(ragResult.params)}
                  </span>
                )}
                {ragResult.evidence_status && (
                  <span className="text-[10px] font-mono px-2.5 py-1 rounded-full"
                    style={{
                      background: noEvidence ? "rgba(245,158,11,0.12)" : "rgba(16,185,129,0.12)",
                      color: noEvidence ? "#b45309" : "#047857",
                      border: `1px solid ${noEvidence ? "rgba(245,158,11,0.3)" : "rgba(16,185,129,0.25)"}`,
                    }}>
                    evidence={ragResult.evidence_status}
                  </span>
                )}
              </div>
            )}

            {noEvidence ? (
              <div className="flex items-start gap-2 px-4 py-3 rounded-xl text-sm"
                style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.3)", color: "#92400e" }}>
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <div>
                  <p className="font-semibold mb-1">No graph evidence</p>
                  <p className="whitespace-pre-wrap">{ragResult.answer}</p>
                  {ragResult.warning && (
                    <p className="mt-2 text-xs font-mono opacity-80">{ragResult.warning}</p>
                  )}
                </div>
              </div>
            ) : (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>Answer</p>
                <p className="text-sm whitespace-pre-wrap leading-relaxed" style={{ color: "var(--text-dark)" }}>{ragResult.answer}</p>
                <p className="text-[11px] mt-2" style={{ color: "var(--text-light)" }}>
                  Grounded in retrieved Neo4j paths from a seeded demo graph — not scientifically validated by CURA.
                </p>
              </div>
            )}

            {ragResult.entities?.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>Entities (click to focus graph)</p>
                <div className="flex flex-wrap gap-2">
                  {ragResult.entities.map((e) => (
                    <button
                      key={e}
                      type="button"
                      onClick={() => onEntityClick(e)}
                      className="px-2.5 py-1 rounded-full text-xs font-medium transition-all hover:-translate-y-0.5"
                      style={{ background: "rgba(63,184,232,0.1)", color: "var(--blue-deep)", border: "1px solid rgba(63,184,232,0.2)" }}
                    >
                      {e}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {ragResult.authors && ragResult.authors.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>
                  Authors (curated AUTHORED_BY)
                </p>
                <div className="flex flex-wrap gap-2">
                  {ragResult.authors.map((a) => (
                    <span
                      key={a}
                      className="px-2.5 py-1 rounded-full text-xs font-medium"
                      style={{ background: "rgba(99,102,241,0.08)", color: "#4338ca", border: "1px solid rgba(99,102,241,0.2)" }}
                    >
                      {a}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {ragResult.paths?.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>
                  Retrieved paths ({ragResult.paths.length})
                </p>
                <ul className="space-y-1.5 max-h-56 overflow-y-auto">
                  {ragResult.paths.slice(0, 40).map((p, i) => (
                    <li key={i} className="text-xs font-mono px-3 py-2 rounded-lg" style={{ background: "#f8fafc", color: "var(--text-mid)" }}>
                      <span style={{ color: "var(--blue-mid)" }}>[{i + 1}]</span> {p}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {ragResult.papers && ragResult.papers.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>
                  Curated papers (PROPOSES / DISCUSSES / RELATED_TO — not clinical proof)
                </p>
                <ul className="space-y-2">
                  {ragResult.papers.map((paper) => (
                    <li key={paper.pmid} className="text-sm px-3 py-2 rounded-xl" style={{ background: "#f8fafc", border: "1px solid var(--border)" }}>
                      <a
                        href={paper.url || `https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-start gap-2 font-medium hover:underline"
                        style={{ color: "var(--blue-deep)" }}
                      >
                        <ExternalLink size={14} className="mt-0.5 shrink-0" />
                        <span>
                          {paper.title || `PMID ${paper.pmid}`}
                          {paper.year ? ` (${paper.year})` : ""}
                          <span className="block text-[11px] font-mono font-normal" style={{ color: "var(--text-light)" }}>
                            PMID {paper.pmid}
                          </span>
                        </span>
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {searchResult && evidenceOnly && (
              <p className="text-[11px] font-mono" style={{ color: "var(--text-light)" }}>
                /graph/search also returned intent={searchResult.intent} with {searchResult.paths?.length ?? 0} catalog paths.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
