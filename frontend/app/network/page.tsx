"use client";

import React, { useState, useEffect } from "react";
import { askGraphRAG, fetchNetwork, type GraphRAGResponse } from "@/lib/api";
import { Network, Loader2, RefreshCcw, Sparkles, AlertCircle } from "lucide-react";

export default function NetworkPage() {
  const [drug, setDrug] = useState("Erlotinib");
  const [disease, setDisease] = useState("Lung cancer");
  const [loading, setLoading] = useState(false);
  const [graphUrl, setGraphUrl] = useState<string | null>(null);
  const [graphError, setGraphError] = useState("");

  const [ragQuery, setRagQuery] = useState(
    "Which drugs share a target with Erlotinib and what diseases are linked?"
  );
  const [ragLoading, setRagLoading] = useState(false);
  const [ragResult, setRagResult] = useState<GraphRAGResponse | null>(null);
  const [ragError, setRagError] = useState("");

  const handleLoad = async () => {
    setLoading(true);
    setGraphError("");
    try {
      const blob = await fetchNetwork(drug, disease);
      const url = URL.createObjectURL(new Blob([blob], { type: "text/html" }));
      setGraphUrl(url);
    } catch (e: unknown) {
      console.error(e);
      setGraphError(e instanceof Error ? e.message : "Failed to load graph");
      setGraphUrl(null);
    } finally {
      setLoading(false);
    }
  };

  const handleGraphRAG = async () => {
    setRagLoading(true);
    setRagError("");
    setRagResult(null);
    try {
      const data = await askGraphRAG(ragQuery);
      setRagResult(data);
    } catch (e: unknown) {
      console.error(e);
      setRagError(e instanceof Error ? e.message : "GraphRAG request failed");
    } finally {
      setRagLoading(false);
    }
  };

  useEffect(() => {
    handleLoad();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-8 flex flex-col" style={{ minHeight: "calc(100vh - 80px)" }}>
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-5">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-widest mb-1" style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}>// Bio Network</p>
          <h1 className="text-4xl font-bold" style={{ color: "var(--text-dark)", letterSpacing: "-1px" }}>Drug–Protein–Disease Network</h1>
          <p className="text-sm mt-1" style={{ color: "var(--text-light)", fontWeight: 300 }}>
            Neo4j knowledge graph visualization + GraphRAG Q&amp;A over multi-hop paths.
          </p>
        </div>

        <div className="flex gap-3 items-end">
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
            onClick={handleLoad}
            disabled={loading}
            className="p-2.5 rounded-xl transition-all hover:-translate-y-0.5 disabled:opacity-50"
            style={{ background: "var(--blue-bright)", color: "#fff", boxShadow: "0 4px 14px rgba(63,184,232,0.3)" }}
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
        <div className="flex items-center gap-2">
          <Sparkles size={18} style={{ color: "var(--blue-mid)" }} />
          <h2 className="text-lg font-semibold" style={{ color: "var(--text-dark)" }}>GraphRAG</h2>
          <span className="text-[10px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full"
            style={{ background: "rgba(63,184,232,0.12)", color: "var(--blue-mid)" }}>
            Neo4j + LLM
          </span>
        </div>
        <p className="text-sm" style={{ color: "var(--text-light)", fontWeight: 300 }}>
          Ask a question; CURA retrieves multi-hop paths from Neo4j and synthesizes an answer grounded in the graph.
        </p>
        <textarea
          value={ragQuery}
          onChange={(e) => setRagQuery(e.target.value)}
          rows={3}
          className="w-full px-4 py-3 rounded-xl text-sm outline-none resize-y"
          style={{ background: "#f8fafc", border: "1px solid var(--border)", color: "var(--text-dark)" }}
          placeholder="e.g. Which drugs share a target with Erlotinib?"
        />
        <button
          onClick={handleGraphRAG}
          disabled={ragLoading || !ragQuery.trim()}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all hover:-translate-y-0.5 disabled:opacity-50"
          style={{ background: "var(--blue-bright)", color: "#fff", boxShadow: "0 4px 14px rgba(63,184,232,0.3)" }}
        >
          {ragLoading ? <Loader2 className="animate-spin" size={16} /> : <Sparkles size={16} />}
          {ragLoading ? "Querying graph…" : "Ask GraphRAG"}
        </button>

        {ragError && (
          <div className="flex items-start gap-2 px-4 py-3 rounded-xl text-sm" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)", color: "#b91c1c" }}>
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span>{ragError}</span>
          </div>
        )}

        {ragResult && (
          <div className="space-y-4 pt-2">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>Answer</p>
              <p className="text-sm whitespace-pre-wrap leading-relaxed" style={{ color: "var(--text-dark)" }}>{ragResult.answer}</p>
            </div>
            {ragResult.entities?.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>Entities</p>
                <div className="flex flex-wrap gap-2">
                  {ragResult.entities.map((e) => (
                    <span key={e} className="px-2.5 py-1 rounded-full text-xs font-medium"
                      style={{ background: "rgba(63,184,232,0.1)", color: "var(--blue-deep)", border: "1px solid rgba(63,184,232,0.2)" }}>
                      {e}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {ragResult.paths?.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>Retrieved paths</p>
                <ul className="space-y-1.5 max-h-48 overflow-y-auto">
                  {ragResult.paths.slice(0, 20).map((p, i) => (
                    <li key={i} className="text-xs font-mono px-3 py-2 rounded-lg" style={{ background: "#f8fafc", color: "var(--text-mid)" }}>
                      {p}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
