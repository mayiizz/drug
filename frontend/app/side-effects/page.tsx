"use client";

import React, { useState } from "react";
import Card from "@/components/Card";
import { predictSideEffects } from "@/lib/api";
import { AlertTriangle, Loader2, ShieldAlert, Heart, Activity, ShieldCheck, AlertCircle } from "lucide-react";

const riskLevel = (effect: string) => {
  if (effect.includes("bleeding") || effect.includes("haemorrhage") || effect.includes("risk"))
    return { bg: "rgba(239,68,68,0.07)", color: "#dc2626", border: "rgba(239,68,68,0.2)", label: "High" };
  if (effect.includes("gastric") || effect.includes("nausea") || effect.includes("vomit"))
    return { bg: "rgba(245,158,11,0.07)", color: "#d97706", border: "rgba(245,158,11,0.2)", label: "Medium" };
  return { bg: "rgba(63,184,232,0.07)", color: "var(--blue-mid)", border: "rgba(63,184,232,0.2)", label: "Low" };
};

export default function SideEffectsPage() {
  const [smiles, setSmiles] = useState("CC(=O)OC1=CC=CC=C1C(=O)O");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!smiles.trim()) {
      setError("Please enter a valid SMILES string.");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await predictSideEffects(smiles);
      if (data.error) {
        setError(data.error);
      } else {
        setResults(data.side_effects);
      }
    } catch (e: any) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.toLowerCase().includes("connect") || msg.toLowerCase().includes("network error")) {
        setError("Invalid SMILES format or prediction failed.");
      } else {
        setError(msg || "Invalid format or model error. Please provide a valid SMILES string.");
      }
    } finally { setLoading(false); }
  };

  const hasHighRisk = results?.some((e) => e.includes("bleeding") || e.includes("haemorrhage"));

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-16">
      <div className="flex items-center gap-4">
        <div className="w-11 h-11 rounded-xl flex items-center justify-center" style={{ background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.2)" }}>
          <ShieldAlert className="w-6 h-6" style={{ color: "#d97706" }} />
        </div>
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-widest mb-1" style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}>// Safety</p>
          <h1 className="text-4xl font-bold" style={{ color: "var(--text-dark)", letterSpacing: "-1px" }}>Side Effect Prediction</h1>
        </div>
      </div>

      <Card title="Safety Assessment" subtitle="Predict potential adverse drug reactions (ADRs) from structural similarity to SIDER database">
        <div className="flex gap-3">
          <input
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handlePredict()}
            className="flex-1 px-4 py-3 rounded-xl text-xs outline-none"
            style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--text-dark)", fontFamily: "var(--font-space-mono)" }}
          />
          <button
            onClick={handlePredict}
            disabled={loading}
            className="px-7 py-3 rounded-xl text-sm font-semibold flex items-center gap-2 transition-all hover:-translate-y-0.5 disabled:opacity-50"
            style={{ background: "#d97706", color: "#fff", boxShadow: "0 4px 14px rgba(245,158,11,0.25)" }}
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <AlertTriangle className="w-4 h-4" />}
            {loading ? "Predicting..." : "Predict ADRs"}
          </button>
        </div>
      </Card>

      {error && (
        <div className="p-4 rounded-xl flex items-start gap-3" style={{ background: "rgba(239,68,68,0.07)", border: "1px solid rgba(239,68,68,0.2)" }}>
          <AlertCircle size={16} className="shrink-0 mt-0.5" style={{ color: "#dc2626" }} />
          <p className="text-sm" style={{ color: "#dc2626" }}>{error}</p>
        </div>
      )}

      {results && results.length > 0 && (
        <>
          {/* Risk notice */}
          <div
            className="p-5 rounded-2xl flex items-start gap-4"
            style={hasHighRisk
              ? { background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)" }
              : { background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.15)" }
            }
          >
            <Heart className="w-7 h-7 shrink-0 animate-pulse" style={{ color: hasHighRisk ? "#dc2626" : "#d97706" }} />
            <div>
              <h4 className="font-semibold text-sm mb-1" style={{ color: "var(--text-dark)" }}>
                {hasHighRisk ? "High Risk Alert" : "Safety Risk Notice"}
              </h4>
              <p className="text-sm leading-relaxed" style={{ color: "var(--text-light)", fontWeight: 300 }}>
                {hasHighRisk
                  ? "Potential bleeding risk detected. Hemostasis monitoring is recommended before clinical evaluation."
                  : "Side effects are predicted from structural similarity to known drugs. These are probabilistic screens and do not replace clinical safety trials."}
              </p>
            </div>
          </div>

          {/* Effect grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {results.map((effect, i) => {
              const r = riskLevel(effect);
              return (
                <div
                  key={i}
                  className="p-4 rounded-xl flex items-center gap-3 transition-all hover:-translate-y-0.5"
                  style={{ background: r.bg, border: `1px solid ${r.border}` }}
                >
                  <Activity className="w-4 h-4 shrink-0" style={{ color: r.color }} />
                  <span className="font-medium text-sm capitalize flex-1" style={{ color: "var(--text-dark)" }}>{effect}</span>
                  <span
                    className="text-[9px] font-bold uppercase px-2 py-0.5 rounded"
                    style={{ color: r.color, background: `${r.bg}` }}
                  >{r.label}</span>
                </div>
              );
            })}
          </div>
        </>
      )}

      {results && results.length === 0 && (
        <div
          className="p-12 text-center rounded-2xl"
          style={{ border: "1px dashed var(--border)", background: "var(--bg)" }}
        >
          <ShieldCheck className="w-12 h-12 mx-auto mb-4" style={{ color: "var(--blue-pale)" }} />
          <h3 className="text-xl font-bold mb-2" style={{ color: "var(--text-dark)" }}>No Structural Alerts Detected</h3>
          <p className="text-sm max-w-md mx-auto" style={{ color: "var(--text-light)", fontWeight: 300 }}>
            No significant side-effect correlations found in the SIDER-linked reference database for this scaffold.
          </p>
        </div>
      )}
    </div>
  );
}
