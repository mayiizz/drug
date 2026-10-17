"use client";

import React, { useState, useEffect } from "react";
import Card from "@/components/Card";
import { discoverMolecules, fetchProtein, Candidate } from "@/lib/api";
import { fetchAIReasoning } from "@/lib/ai";
import { calculateSAScore, getSAColor } from "@/lib/saScore";
import { Sparkles, Loader2, Dna, AlertCircle, ChevronDown, ChevronUp, Zap, Microscope, Download } from "lucide-react";
import { useSearchParams, useRouter } from "next/navigation";
import { formatKd } from "@/lib/format";

const ScorePill = ({ label, value, color }: { label: string; value: string; color: string }) => (
  <div className="flex flex-col items-center px-3 py-2 rounded-xl" style={{ background: `${color}10`, border: `1px solid ${color}25` }}>
    <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color, fontFamily: "var(--font-space-mono)" }}>{label}</span>
    <span className="text-base font-bold mt-0.5" style={{ color: "var(--text-dark)" }}>{value}</span>
  </div>
);

function CandidateCard({ c, rank, target, seedSmiles }: { c: Candidate; rank: number; target: string; seedSmiles: string }) {
  const [expanded, setExpanded] = useState(rank === 0);
  const [reasoning, setReasoning] = useState<string | null>(null);
  const [loadingAI, setLoadingAI] = useState(false);
  const router = useRouter();
  const sa = calculateSAScore(c.smiles);
  const saStyle = getSAColor(sa);

  const loadReasoning = async () => {
    if (reasoning) return;
    setLoadingAI(true);
    try {
      const text = await fetchAIReasoning({
        target, smiles: c.smiles,
        qed: c.drug_likeness,
        logp: 2.5, tpsa: 90,
        mw: 400,
        affinity: c.binding_score,
        sa,
      });
      setReasoning(text);
    } catch { setReasoning("AI reasoning unavailable."); }
    finally { setLoadingAI(false); }
  };

  useEffect(() => { if (rank === 0) loadReasoning(); }, []);

  const verdict = c.validation?.verdict;
  const verdictStyle = verdict === "PASS"
    ? { bg: "rgba(16,185,129,0.08)", color: "#059669", border: "rgba(16,185,129,0.2)" }
    : verdict === "WARN"
      ? { bg: "rgba(245,158,11,0.08)", color: "#d97706", border: "rgba(245,158,11,0.2)" }
      : { bg: "rgba(239,68,68,0.08)", color: "#dc2626", border: "rgba(239,68,68,0.2)" };

  return (
    <Card className="overflow-hidden">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center font-bold text-xs shrink-0"
            style={{ background: rank === 0 ? "var(--blue-bright)" : "var(--bg)", color: rank === 0 ? "#fff" : "var(--text-light)", fontFamily: "var(--font-space-mono)" }}
          >
            #{rank + 1}
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest mb-0.5" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>Candidate</p>
            <p className="text-xs font-mono break-all" style={{ color: "var(--blue-mid)" }}>{c.smiles.slice(0, 40)}{c.smiles.length > 40 ? "…" : ""}</p>
          </div>
        </div>
        {verdict && (
          <span
            className="px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider shrink-0"
            style={{ background: verdictStyle.bg, color: verdictStyle.color, border: `1px solid ${verdictStyle.border}` }}
          >
            {verdict}
          </span>
        )}
      </div>

      {/* Scores */}
      <div className="flex gap-2 flex-wrap mb-4">
        <ScorePill label="QED" value={(c.drug_likeness ?? 0).toFixed(2)} color="var(--blue-bright)" />
        <ScorePill label="Kd" value={formatKd(c.binding_score)} color="#8b5cf6" />
        <ScorePill label="SA" value={sa.toFixed(1)} color={saStyle.color.replace("text-", "").replace("-400", "")} />
        {c.validation?.composite_score != null && (
          <ScorePill label="Score" value={(c.validation.composite_score * 100).toFixed(0) + "%"} color="var(--blue-mid)" />
        )}
      </div>

      {/* AI Reasoning */}
      <button
        onClick={() => { setExpanded(!expanded); if (!expanded) loadReasoning(); }}
        className="w-full flex items-center justify-between p-3 rounded-xl transition-all text-sm font-medium"
        style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--text-mid)" }}
      >
        <span className="flex items-center gap-2"><Zap size={13} style={{ color: "var(--blue-mid)" }} /> AI Reasoning</span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {expanded && (
        <div className="mt-3 p-4 rounded-xl text-sm leading-relaxed" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          {loadingAI ? (
            <div className="flex items-center gap-2" style={{ color: "var(--text-light)" }}>
              <Loader2 size={14} className="animate-spin" /> Generating reasoning...
            </div>
          ) : (
            <p className="whitespace-pre-line text-xs" style={{ color: "var(--text-mid)", fontWeight: 300, lineHeight: 1.75 }}>
              {reasoning}
            </p>
          )}
        </div>
      )}

      {/* Validation params */}
      {c.validation?.parameters && (
        <div className="mt-3 grid grid-cols-3 gap-1.5">
          {Object.entries(c.validation.parameters).slice(0, 6).map(([k, v]) => (
            <div key={k} className="p-2 rounded-lg text-center" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
              <p className="text-[9px] uppercase font-semibold" style={{ color: "var(--text-light)" }}>{k.replace(/_/g, " ")}</p>
              <p className="text-xs font-bold mt-0.5" style={{
                color: v.verdict === "PASS" ? "#059669" : v.verdict === "WARN" ? "#d97706" : "#dc2626"
              }}>{typeof v.value === "number" ? v.value.toFixed(2) : v.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Analyze Button */}
      <button
        onClick={() => router.push(`/analysis?smiles=${encodeURIComponent(c.smiles)}`)}
        className="mt-4 w-full py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all hover:-translate-y-0.5"
        style={{
          background: "transparent",
          border: "1px solid var(--blue-mid)",
          color: "var(--blue-mid)",
        }}
      >
        <Microscope size={14} />
        Analyze Molecule
      </button>
    </Card>
  );
}

export default function DiscoveryPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [smiles, setSmiles] = useState("CC(=O)Nc1ccc(O)cc1");
  const [target, setTarget] = useState(searchParams.get("target") ?? "EGFR");
  const [protein, setProtein] = useState<{ sequence: string; name: string; uniprot_id: string } | null>(null);
  const [loadingProtein, setLoadingProtein] = useState(false);
  const [loading, setLoading] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleFetchProtein = async () => {
    if (!target.trim()) return;
    setLoadingProtein(true);
    setError(null);
    try {
      const data = await fetchProtein(target);
      if (data.error) { setError(data.error); setProtein(null); }
      else setProtein(data);
    } catch (e: any) { setError(e instanceof Error ? e.message : "Could not fetch protein. Check the gene name."); }
    finally { setLoadingProtein(false); }
  };

  const handleDiscover = async () => {
    if (!protein?.sequence) { setError("Please fetch the protein target first."); return; }
    if (!smiles.trim()) { setError("Please enter a seed SMILES."); return; }
    setLoading(true); setError(null); setCandidates([]);
    try {
      const data = await discoverMolecules(smiles, target, protein.sequence);
      if (data.error) { setError(data.error); }
      else setCandidates(data.candidates ?? []);
    } catch (e: any) {
      // api.ts interceptor converts ALL axios errors to plain Error with human-readable message
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg || "Discovery failed. Please check your inputs and try again.");
    } finally { setLoading(false); }
  };

  // auto-trigger from copilot
  useEffect(() => {
    if (searchParams.get("trigger") === "true" && target) handleFetchProtein();
  }, []);

  const handleExportCSV = () => {
    if (candidates.length === 0) return;
    const header = "SMILES,QED,Binding_nM,SA_Score,Total_Score\n";
    const rows = candidates.map((c) => {
      const sa = calculateSAScore(c.smiles);
      const score = c.validation?.composite_score != null ? c.validation.composite_score * 100 : "";
      return `${c.smiles},${c.drug_likeness ?? ""},${c.binding_score ?? ""},${sa.toFixed(2)},${score}`;
    }).join("\n");
    
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `discovery_${target}_results.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportSDF = () => {
    if (candidates.length === 0) return;
    let sdf = "";
    candidates.forEach((c, index) => {
      const sa = calculateSAScore(c.smiles);
      const score = c.validation?.composite_score != null ? c.validation.composite_score * 100 : "";
      sdf += `${c.smiles} Candidate_${index + 1}\n`;
      sdf += `>  <QED>\n${c.drug_likeness ?? ""}\n\n`;
      sdf += `>  <Binding_nM>\n${c.binding_score ?? ""}\n\n`;
      sdf += `>  <SA_Score>\n${sa.toFixed(2)}\n\n`;
      sdf += `>  <Total_Score>\n${score}\n\n`;
      sdf += `$$$$\n`;
    });
    const blob = new Blob([sdf], { type: "chemical/x-mdl-sdfile" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `discovery_${target}_results.sdf`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-16">
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}>// Drug Generation with seed molecule</p>
        <h1 className="text-4xl font-bold mb-2" style={{ color: "var(--text-dark)", letterSpacing: "-1px" }}>Drug Generation</h1>
        <p className="text-sm" style={{ color: "var(--text-light)", fontWeight: 300 }}>
          MolGPT generation with mandatory DeepPurpose binding pre-check. Protein sequence required.
        </p>
      </div>

      {/* Step 1 — Protein */}
      <Card title="Step 1 · Fetch Protein Target" subtitle="Gene name or UniProt ID — required for binding validation">
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Dna size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2" style={{ color: "var(--blue-mid)" }} />
            <input
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleFetchProtein()}
              placeholder="e.g. EGFR, HER2, TP53"
              className="w-full pl-9 pr-4 py-3 rounded-xl text-sm outline-none"
              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--text-dark)" }}
            />
          </div>
          <button
            onClick={handleFetchProtein}
            disabled={loadingProtein}
            className="px-6 py-3 rounded-xl text-sm font-semibold flex items-center gap-2 transition-all hover:-translate-y-0.5 disabled:opacity-50"
            style={{ background: "var(--text-dark)", color: "#fff" }}
          >
            {loadingProtein ? <Loader2 className="w-4 h-4 animate-spin" /> : <Dna className="w-4 h-4" />}
            {loadingProtein ? "Fetching..." : "Fetch Protein"}
          </button>
        </div>

        {protein && (
          <div className="mt-4 p-4 rounded-xl flex items-start gap-3" style={{ background: "rgba(63,184,232,0.06)", border: "1px solid rgba(63,184,232,0.2)" }}>
            <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0" style={{ background: "rgba(63,184,232,0.15)" }}>
              <Dna size={14} style={{ color: "var(--blue-mid)" }} />
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-sm" style={{ color: "var(--text-dark)" }}>{protein.name} · {protein.uniprot_id}</p>
              <p className="text-xs mt-1 truncate" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>
                {protein.sequence.slice(0, 80)}…
              </p>
            </div>
            <span className="px-2 py-1 rounded-lg text-[10px] font-bold shrink-0" style={{ background: "rgba(16,185,129,0.1)", color: "#059669", border: "1px solid rgba(16,185,129,0.2)" }}>
              VERIFIED
            </span>
          </div>
        )}
      </Card>

      {/* Step 2 — Seed + Generate */}
      <Card
        title="Step 2 · Provide Seed Drug & Generate"
        subtitle="SMILES of a known drug for this target. Binding must be < 1000 nM."
      >
        <div className="flex gap-3">
          <input
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            placeholder="Seed SMILES — must bind your target protein"
            className="flex-1 px-4 py-3 rounded-xl text-xs outline-none"
            style={{
              background: "var(--bg)", border: "1px solid var(--border)",
              color: "var(--text-dark)", fontFamily: "var(--font-space-mono)",
            }}
          />
          <button
            onClick={handleDiscover}
            disabled={loading || !protein}
            className="px-7 py-3 rounded-xl text-sm font-semibold flex items-center gap-2 transition-all hover:-translate-y-0.5 disabled:opacity-50"
            style={{ background: "var(--blue-bright)", color: "#fff", boxShadow: "0 4px 14px rgba(63,184,232,0.3)" }}
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {loading ? "Generating..." : "Generate Candidates"}
          </button>
        </div>
        {!protein && (
          <p className="text-xs mt-3 flex items-center gap-1.5" style={{ color: "var(--text-light)" }}>
            <AlertCircle size={12} /> Fetch a protein target first to enable generation.
          </p>
        )}
      </Card>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-xl flex items-start gap-3" style={{ background: "rgba(239,68,68,0.07)", border: "1px solid rgba(239,68,68,0.2)" }}>
          <AlertCircle size={16} className="shrink-0 mt-0.5" style={{ color: "#dc2626" }} />
          <p className="text-sm" style={{ color: "#dc2626" }}>{error}</p>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="text-center py-16">
          <div className="w-12 h-12 rounded-full border-2 animate-spin mx-auto mb-4"
            style={{ borderColor: "var(--border)", borderTopColor: "var(--blue-bright)" }} />
          <p className="font-medium text-sm" style={{ color: "var(--text-mid)" }}>Running MolGPT generation + binding validation…</p>
          <p className="text-xs mt-1" style={{ color: "var(--text-light)" }}>This may take 1–2 minutes</p>
        </div>
      )}

      {/* Candidates */}
      {candidates.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-xl font-bold" style={{ color: "var(--text-dark)", letterSpacing: "-0.5px" }}>
              {candidates.length} Candidates · Ranked by Binding
            </h2>
            <span className="text-xs px-3 py-1.5 rounded-full font-semibold" style={{ background: "rgba(63,184,232,0.1)", color: "var(--blue-mid)", border: "1px solid rgba(63,184,232,0.2)" }}>
              Target: {target}
            </span>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {candidates.map((c, i) => (
              <CandidateCard key={i} c={c} rank={i} target={target} seedSmiles={smiles} />
            ))}
          </div>

          {/* Export Buttons */}
          <div className="flex justify-center gap-4 pt-4 mt-6" style={{ borderTop: "1px solid var(--border)" }}>
            <button
              onClick={handleExportCSV}
              className="px-6 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all hover:-translate-y-0.5"
              style={{
                background: "transparent",
                border: "1px solid var(--border)",
                color: "var(--text-dark)",
              }}
            >
              <Download size={14} />
              Export CSV
            </button>
            <button
              onClick={handleExportSDF}
              className="px-6 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all hover:-translate-y-0.5"
              style={{
                background: "transparent",
                border: "1px solid var(--border)",
                color: "var(--text-dark)",
              }}
            >
              <Download size={14} />
              Export SDF
            </button>
          </div>
        </div>
      )}
    </div>
  );
}