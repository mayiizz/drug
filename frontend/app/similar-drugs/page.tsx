"use client";

import React, { useState } from "react";
import Card from "@/components/Card";
import { findSimilarDrugs } from "@/lib/api";
import { Search, Loader2, Microscope, Info, AlertCircle } from "lucide-react";
import { useRouter } from "next/navigation";

export default function SimilarDrugsPage() {
  const router = useRouter();
  const [smiles, setSmiles] = useState("CC(=O)OC1=CC=CC=C1C(=O)O");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!smiles.trim()) {
      setError("Please enter a valid SMILES string.");
      return;
    }
    setLoading(true);
    setError(null);
    setResults([]);
    try {
      const data = await findSimilarDrugs(smiles);
      if (data.error) {
        setError(data.error);
      } else {
        setResults(data.similar_drugs ?? []);
      }
    } catch (e: any) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.toLowerCase().includes("connect") || msg.toLowerCase().includes("network error")) {
        setError("Invalid SMILES format or no similar drugs found in database.");
      } else {
        setError(msg || "Invalid format or molecule not found in database.");
      }
    } finally { setLoading(false); }
  };

  const getInsight = (score: number) => {
    if (score > 0.8) return "Strong structural overlay — nearly identical sub-graphs and functional groups.";
    if (score > 0.5) return "Significant backbone similarity — key pharmacokinetic features preserved.";
    return "Broad structural resemblance — common essential fragments, different side chains.";
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-16">
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}>// Drug Intelligence</p>
        <h1 className="text-4xl font-bold mb-2" style={{ color: "var(--text-dark)", letterSpacing: "-1px" }}>Similar Drug Finder</h1>
        <p className="text-sm" style={{ color: "var(--text-light)", fontWeight: 300 }}>Discover existing drugs with similar structural embeddings using fingerprint similarity.</p>
      </div>

      <Card title="Reference Molecule" subtitle="Enter the SMILES to search against">
        <div className="flex gap-3">
          <input
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            className="flex-1 px-4 py-3 rounded-xl text-xs outline-none"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--text-dark)", fontFamily: "var(--font-space-mono)" }}
          />
          <button
            onClick={handleSearch}
            disabled={loading}
            className="px-7 py-3 rounded-xl text-sm font-semibold flex items-center gap-2 transition-all hover:-translate-y-0.5 disabled:opacity-50"
            style={{ background: "var(--blue-bright)", color: "#fff", boxShadow: "0 4px 14px rgba(63,184,232,0.3)" }}
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </Card>

      {error && (
        <div className="p-4 rounded-xl flex items-start gap-3" style={{ background: "rgba(239,68,68,0.07)", border: "1px solid rgba(239,68,68,0.2)" }}>
          <AlertCircle size={16} className="shrink-0 mt-0.5" style={{ color: "#dc2626" }} />
          <p className="text-sm" style={{ color: "#dc2626" }}>{error}</p>
        </div>
      )}

      {results.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {results.map((drug, i) => {
            const score = drug.score ?? (1 - i * 0.05);
            return (
              <Card key={i} className="flex flex-col h-full">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1 min-w-0 mr-3">
                    <h3 className="font-semibold text-sm truncate" style={{ color: "var(--text-dark)" }}>{drug.name}</h3>
                    <p className="text-[10px] mt-1 truncate" style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}>{drug.smiles}</p>
                  </div>
                  <span
                    className="px-2.5 py-1 rounded-lg text-[10px] font-bold shrink-0"
                    style={{ background: "rgba(63,184,232,0.1)", color: "var(--blue-mid)", border: "1px solid rgba(63,184,232,0.2)" }}
                  >
                    {(score * 100).toFixed(0)}%
                  </span>
                </div>

                {/* Match bar */}
                <div className="mb-4">
                  <div className="h-1.5 w-full rounded-full" style={{ background: "var(--bg)" }}>
                    <div className="h-full rounded-full transition-all" style={{ width: `${score * 100}%`, background: "var(--blue-bright)" }} />
                  </div>
                </div>

                <div
                  className="flex-1 p-3 rounded-xl mb-4"
                  style={{ background: "var(--bg)", border: "1px solid var(--border)" }}
                >
                  <div className="flex items-center gap-1.5 mb-1.5" style={{ color: "var(--text-light)" }}>
                    <Info size={11} />
                    <span className="text-[9px] uppercase font-bold tracking-wider">Similarity Insight</span>
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--text-mid)", fontWeight: 300 }}>
                    {getInsight(score)}
                  </p>
                </div>

                <button
                  onClick={() => router.push(`/analysis?smiles=${encodeURIComponent(drug.smiles)}`)}
                  className="w-full py-2.5 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-all hover:-translate-y-0.5 mt-auto"
                  style={{ background: "rgba(63,184,232,0.08)", color: "var(--blue-mid)", border: "1px solid rgba(63,184,232,0.2)" }}
                >
                  <Microscope size={14} /> Analyze
                </button>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
