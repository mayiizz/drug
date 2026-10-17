"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Card from "@/components/Card";
import {
  FlaskConical, Loader2, AlertCircle, CheckCircle2,
  ChevronDown, ChevronUp, Sparkles, Atom, Dna,
  Activity, Zap, Info, Microscope, Download
} from "lucide-react";
import { formatKd } from "@/lib/format";

// ── API ───────────────────────────────────────────────────────────────────────
async function runDeNovo(
  gene_name: string,
  num_molecules: number,
  sample_size: number
): Promise<any> {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
  const res = await fetch(`${apiBase}/denovo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gene_name, num_molecules, sample_size }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail ?? `Server error ${res.status}`);
  return data;
}

// ── Validation ────────────────────────────────────────────────────────────────
function validateGene(g: string): string | null {
  if (!g) return "Please enter a gene name or UniProt accession (e.g. EGFR, BRAF, CDK4).";
  if (g.length < 2) return "Gene name is too short. Use a valid symbol like EGFR.";
  if (g.length > 20) return "Gene name is too long. Use a short symbol like EGFR, not the full protein name.";
  if (!/^[A-Za-z][A-Za-z0-9]{1,19}$/.test(g))
    return "Gene name must start with a letter and contain only letters and numbers (e.g. EGFR, CDK4, TP53).";
  return null;
}

function parseBackendError(msg: string, gene: string): string {
  if (msg.includes("not found in UniProt") || msg.includes("not a valid gene") || msg.includes("404"))
    return `"${gene}" was not found in UniProt. Use an official HGNC gene symbol (EGFR, BRAF, CDK4, TP53, HER2, KRAS, ACE2) or a UniProt accession like P00533.`;
  if (msg.includes("too short") || msg.includes("too small"))
    return "The protein sequence returned is too short. Try a different gene name or UniProt accession.";
  if (msg.includes("No drug-like molecules") || msg.includes("no candidates"))
    return "No drug-like molecules were found. Try increasing the sample size, or try a well-studied target like EGFR or BRAF.";
  if (msg.includes("bind") && msg.includes("None"))
    return `None of the generated molecules showed sufficient binding affinity for ${gene}. Try sample size 800–1000 for broader coverage.`;
  if (msg.includes("REINVENT") || msg.includes("smiles_lstm_prior"))
    return "REINVENT prior checkpoint not found. Checked models/generation/reinvent4/weights and models/generation/weights for smiles_lstm_prior.pt on the server.";
  if (msg.includes("3D geometry") || msg.includes("conformer"))
    return "All generated molecules failed 3D geometry validation. Try a larger sample size.";
  if (msg.includes("ESMFold") || msg.includes("fold"))
    return "Protein structure prediction failed. The ESMFold API may be temporarily unavailable — the pipeline will use fallback seeds.";
  if (msg.includes("connect") || msg.includes("fetch") || msg.includes("ECONNREFUSED"))
    return "Cannot connect to the backend server. Make sure it is running on port 8000.";
  return msg;
}

// ── Candidate card ────────────────────────────────────────────────────────────
function CandidateCard({ c, index }: { c: any; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const router = useRouter();

  const isPass = c.status === "PASS";
  const statusStyle = isPass
    ? { bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.25)", text: "#059669" }
    : { bg: "rgba(245,158,11,0.08)", border: "rgba(245,158,11,0.25)", text: "#d97706" };

  const scoreColor = (val: number, good: number, ok: number) =>
    val >= good ? "#059669" : val >= ok ? "#d97706" : "#dc2626";

  return (
    <div
      className="rounded-2xl overflow-hidden transition-all hover:shadow-lg"
      style={{
        background: "var(--card-bg)",
        border: "1px solid var(--border)",
        transition: "box-shadow 0.2s",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 pt-4 pb-3">
        <div className="flex items-center gap-2.5">
          <span
            className="text-xs font-bold px-2.5 py-1 rounded-lg"
            style={{
              background: "rgba(63,184,232,0.1)",
              color: "var(--blue-mid)",
              fontFamily: "var(--font-space-mono)",
            }}
          >
            #{index + 1}
          </span>
          <span
            className="text-xs font-semibold uppercase tracking-wider"
            style={{ color: "var(--text-light)" }}
          >
            Candidate Drug
          </span>
        </div>
        <span
          className="text-xs font-bold px-2.5 py-1 rounded-lg"
          style={{
            background: statusStyle.bg,
            border: `1px solid ${statusStyle.border}`,
            color: statusStyle.text,
          }}
        >
          {c.status}
        </span>
      </div>

      {/* SMILES */}
      <div className="px-5 pb-3">
        <p
          className="text-[11px] font-mono break-all leading-relaxed px-3 py-2.5 rounded-xl select-all"
          style={{
            background: "var(--bg)",
            color: "var(--blue-deep)",
            border: "1px solid var(--border)",
            cursor: "text",
          }}
          title="Click to select SMILES"
        >
          {c.smiles}
        </p>
      </div>

      {/* Score chips */}
      <div className="px-5 pb-3 grid grid-cols-4 gap-2">
        {[
          {
            label: "QED",
            value: c.qed?.toFixed(2) ?? "—",
            color: scoreColor(c.qed ?? 0, 0.6, 0.4),
            tooltip: "Drug-likeness (0–1, higher is better)",
          },
          {
            label: "Kd",
            value: c.binding_nm != null ? formatKd(c.binding_nm) : "—",
            color: scoreColor(10000 - (c.binding_nm ?? 9999), 9000, 5000),
            tooltip: "Predicted dissociation constant (lower = stronger)",
          },
          {
            label: "SA",
            value: c.sa_score?.toFixed(1) ?? "—",
            color: scoreColor(10 - (c.sa_score ?? 10), 6, 4),
            tooltip: "Synthetic accessibility (1=easy, 10=hard)",
          },
          {
            label: "Score",
            value: c.score != null ? `${c.score.toFixed(1)}%` : "—",
            color: scoreColor(c.score ?? 0, 60, 40),
            tooltip: "Composite ranking score",
          },
        ].map(({ label, value, color, tooltip }) => (
          <div
            key={label}
            className="flex flex-col items-center px-2 py-2 rounded-xl"
            style={{ background: "var(--bg)", border: "1px solid var(--border)" }}
            title={tooltip}
          >
            <span
              className="text-[9px] font-semibold uppercase tracking-widest mb-0.5"
              style={{ color: "var(--text-light)" }}
            >
              {label}
            </span>
            <span className="text-sm font-bold" style={{ color }}>
              {value}
            </span>
          </div>
        ))}
      </div>

      {/* Reasoning */}
      {c.reasoning && (
        <div className="px-5 pb-3">
          <p
            className="text-xs leading-relaxed px-3 py-2 rounded-xl"
            style={{
              color: "var(--text-mid)",
              background: "var(--bg)",
              border: "1px solid var(--border)",
            }}
          >
            {c.reasoning}
          </p>
        </div>
      )}

      {/* Expand toggle */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-2.5 text-xs font-semibold transition-colors"
        style={{
          borderTop: "1px solid var(--border)",
          color: "var(--text-light)",
          background: expanded ? "var(--bg)" : "transparent",
        }}
      >
        <span>Molecular properties</span>
        {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>

      {expanded && (
        <div className="px-5 pb-4 grid grid-cols-3 gap-2 pt-3">
          {[
            { label: "Mol Weight", value: c.mol_weight != null ? `${c.mol_weight.toFixed(1)} Da` : "—" },
            { label: "LogP", value: c.logp?.toFixed(2) ?? "—" },
            { label: "TPSA", value: c.tpsa != null ? `${c.tpsa.toFixed(1)} Å²` : "—" },
            { label: "Rot Bonds", value: c.rot_bonds ?? "—" },
            { label: "H Donors", value: c.h_donors ?? "—" },
            { label: "H Acceptors", value: c.h_acceptors ?? "—" },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="flex flex-col px-3 py-2 rounded-xl"
              style={{ background: "var(--bg)", border: "1px solid var(--border)" }}
            >
              <span
                className="text-[9px] font-semibold uppercase tracking-widest mb-0.5"
                style={{ color: "var(--text-light)" }}
              >
                {label}
              </span>
              <span className="text-sm font-bold" style={{ color: "var(--text-dark)" }}>
                {value}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Analyze Button */}
      <div className="px-5 pb-4 pt-1">
        <button
          onClick={() => router.push(`/analysis?smiles=${encodeURIComponent(c.smiles)}`)}
          className="w-full py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all hover:-translate-y-0.5"
          style={{
            background: "transparent",
            border: "1px solid var(--blue-mid)",
            color: "var(--blue-mid)",
          }}
        >
          <Microscope size={14} />
          Analyze Molecule
        </button>
      </div>
    </div>
  );
}

// ── Error banner ──────────────────────────────────────────────────────────────
function ErrorBanner({ message }: { message: string }) {
  return (
    <div
      className="flex items-start gap-3 p-4 rounded-xl text-sm"
      style={{
        background: "rgba(239,68,68,0.07)",
        border: "1px solid rgba(239,68,68,0.2)",
        color: "#dc2626",
      }}
    >
      <AlertCircle size={16} className="mt-0.5 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

// ── Success banner ────────────────────────────────────────────────────────────
function SuccessBanner({ message }: { message: string }) {
  return (
    <div
      className="flex items-start gap-3 p-4 rounded-xl text-sm"
      style={{
        background: "rgba(16,185,129,0.07)",
        border: "1px solid rgba(16,185,129,0.2)",
        color: "#059669",
      }}
    >
      <CheckCircle2 size={16} className="mt-0.5 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

// ── Pipeline step indicator ───────────────────────────────────────────────────
function PipelineSteps({ sample }: { sample: number }) {
  const steps = [
    { icon: Dna, label: "UniProt fetch", desc: "Fetching protein sequence" },
    { icon: Atom, label: "ESMFold", desc: "Predicting 3D structure" },
    { icon: Activity, label: "Pocket extraction", desc: "Finding binding site" },
    { icon: Sparkles, label: "REINVENT generation", desc: `Sampling ${sample} molecules` },
    { icon: Zap, label: "Filtering & ranking", desc: "Binding · ADMET · QED" },
  ];

  return (
    <div
      className="p-4 rounded-xl space-y-3"
      style={{
        background: "rgba(63,184,232,0.04)",
        border: "1px solid rgba(63,184,232,0.15)",
      }}
    >
      <p className="text-xs font-semibold" style={{ color: "var(--blue-mid)" }}>
        Pipeline running — this may take 1–3 minutes
      </p>
      <div className="space-y-2">
        {steps.map(({ icon: Icon, label, desc }) => (
          <div key={label} className="flex items-center gap-3">
            <div
              className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0"
              style={{ background: "rgba(63,184,232,0.12)", border: "1px solid rgba(63,184,232,0.2)" }}
            >
              <Icon size={11} style={{ color: "var(--blue-mid)" }} />
            </div>
            <div className="flex-1 min-w-0">
              <span className="text-xs font-semibold" style={{ color: "var(--text-dark)" }}>
                {label}
              </span>
              <span className="text-xs ml-2" style={{ color: "var(--text-light)" }}>
                {desc}
              </span>
            </div>
            <Loader2 size={11} className="animate-spin shrink-0" style={{ color: "var(--blue-mid)" }} />
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function DeNovoPage() {
  const [gene, setGene] = useState("EGFR");
  const [numMol, setNumMol] = useState(10);
  const [sample, setSample] = useState(500);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [fieldErr, setFieldErr] = useState<string | null>(null);

  const handleGenerate = async () => {
    const g = gene.trim().toUpperCase();

    // Client-side validation
    const validationErr = validateGene(g);
    if (validationErr) {
      setFieldErr(validationErr);
      return;
    }

    setFieldErr(null);
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const data = await runDeNovo(g, numMol, sample);
      setResult(data);
    } catch (e: any) {
      setError(parseBackendError(e.message ?? "Unknown error", g));
    } finally {
      setLoading(false);
    }
  };

  const handleGeneChange = (v: string) => {
    setGene(v);
    setFieldErr(null);
    setError(null);
  };

  const handleExportCSV = () => {
    if (!result?.candidates) return;
    const header = "SMILES,QED,Binding_nM,SA_Score,Total_Score\n";
    const rows = result.candidates.map((c: any) =>
      `${c.smiles},${c.qed ?? ""},${c.binding_nm ?? ""},${c.sa_score ?? ""},${c.score ?? ""}`
    ).join("\n");

    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `denovo_${result.target}_results.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportSDF = () => {
    if (!result?.candidates) return;
    let sdf = "";
    result.candidates.forEach((c: any, index: number) => {
      sdf += `${c.smiles} Candidate_${index + 1}\n`;
      sdf += `>  <QED>\n${c.qed}\n\n`;
      sdf += `>  <Binding_nM>\n${c.binding_nm}\n\n`;
      sdf += `>  <SA_Score>\n${c.sa_score}\n\n`;
      sdf += `>  <Total_Score>\n${c.score}\n\n`;
      sdf += `$$$$\n`;
    });
    const blob = new Blob([sdf], { type: "chemical/x-mdl-sdfile" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `denovo_${result.target}_results.sdf`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-16">

      {/* Page header */}
      <div className="flex items-center gap-4">
        <div
          className="w-11 h-11 rounded-xl flex items-center justify-center"
          style={{ background: "rgba(63,184,232,0.1)", border: "1px solid rgba(63,184,232,0.2)" }}
        >
          <FlaskConical className="w-6 h-6" style={{ color: "var(--blue-mid)" }} />
        </div>
        <div>
          <p
            className="text-[10px] font-semibold uppercase tracking-widest mb-1"
            style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}
          >
            Protein-Based Drug Generation
          </p>
          <h1
            className="text-4xl font-bold"
            style={{ color: "var(--text-dark)", letterSpacing: "-1px" }}
          >
            Generate Drug Candidates
          </h1>
        </div>
      </div>

      {/* How it works info strip */}
      <div
        className="flex items-start gap-3 p-4 rounded-xl text-xs"
        style={{
          background: "rgba(63,184,232,0.04)",
          border: "1px solid rgba(63,184,232,0.12)",
          color: "var(--text-mid)",
        }}
      >
        <Info size={14} className="mt-0.5 shrink-0" style={{ color: "var(--blue-mid)" }} />
        <span>
          Enter a protein target and CURA will automatically fetch its sequence from UniProt,
          predict its 3D structure with <strong>ESMFold</strong>, extract the binding pocket,
          and generate novel drug candidates using <strong>pocket-guided REINVENT</strong> — no seed molecule required.
        </span>
      </div>

      {/* Main form */}
      <Card
        title="Protein-Based Drug Generation"
        subtitle="Generate novel drug candidates from a protein target alone — no seed molecule required"
      >
        <div className="space-y-5 mt-1">

          {/* Gene input */}
          <div>
            <label
              className="text-[10px] font-semibold uppercase tracking-widest block mb-2"
              style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}
            >
              Target Gene / UniProt ID
            </label>
            <input
              value={gene}
              onChange={(e) => handleGeneChange(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !loading && handleGenerate()}
              placeholder="e.g. EGFR, BRAF, CDK4, TP53, HER2, KRAS, ACE2"
              disabled={loading}
              className="w-full px-4 py-3 rounded-xl text-sm font-semibold outline-none transition-all"
              style={{
                background: "var(--bg)",
                border: `1px solid ${fieldErr ? "rgba(239,68,68,0.5)" : "var(--border)"}`,
                color: "var(--text-dark)",
                opacity: loading ? 0.6 : 1,
              }}
            />
            {/* Inline field error */}
            {fieldErr ? (
              <p className="text-[11px] mt-1.5 flex items-center gap-1.5" style={{ color: "#dc2626" }}>
                <AlertCircle size={11} />
                {fieldErr}
              </p>
            ) : (
              <p className="text-[11px] mt-1.5" style={{ color: "var(--text-light)" }}>
                Must be an official HGNC gene symbol or UniProt accession. Protein fetched automatically from UniProt.
              </p>
            )}
          </div>

          {/* Controls */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label
                className="text-[10px] font-semibold uppercase tracking-widest block mb-2"
                style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}
              >
                Candidates to return
              </label>
              <select
                value={numMol}
                onChange={(e) => setNumMol(Number(e.target.value))}
                disabled={loading}
                className="w-full px-4 py-3 rounded-xl text-sm font-semibold outline-none"
                style={{
                  background: "var(--bg)",
                  border: "1px solid var(--border)",
                  color: "var(--text-dark)",
                  opacity: loading ? 0.6 : 1,
                }}
              >
                {[1, 2, 3, 5, 8, 10].map((n) => (
                  <option key={n} value={n}>
                    {n} candidate{n > 1 ? "s" : ""}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label
                className="text-[10px] font-semibold uppercase tracking-widest block mb-2"
                style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}
              >
                Sample size
              </label>
              <select
                value={sample}
                onChange={(e) => setSample(Number(e.target.value))}
                disabled={loading}
                className="w-full px-4 py-3 rounded-xl text-sm font-semibold outline-none"
                style={{
                  background: "var(--bg)",
                  border: "1px solid var(--border)",
                  color: "var(--text-dark)",
                  opacity: loading ? 0.6 : 1,
                }}
              >
                <option value={100}>100 — fast </option>
                <option value={200}>200 — balanced</option>
                <option value={500}>500 — recommended </option>
                <option value={800}>800 — thorough </option>
                <option value={1000}>1000 — maximum </option>
              </select>
              <p className="text-[11px] mt-1.5" style={{ color: "var(--text-light)" }}>
                Higher sample size = more diverse candidates, longer wait.
              </p>
            </div>
          </div>

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full py-3.5 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-all hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
            style={{
              background: "var(--blue-bright)",
              color: "#fff",
              boxShadow: loading ? "none" : "0 4px 14px rgba(63,184,232,0.3)",
            }}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating candidates…
              </>
            ) : (
              <>
                <FlaskConical className="w-4 h-4" />
                Generate Drug Candidates
              </>
            )}
          </button>

          {/* Pipeline progress */}
          {loading && <PipelineSteps sample={sample} />}
        </div>
      </Card>

      {/* Error */}
      {error && !loading && <ErrorBanner message={error} />}

      {/* Results */}
      {result && !loading && (
        <div className="space-y-5">
          <SuccessBanner
            message={`Generated ${result.total_returned} candidate${result.total_returned !== 1 ? "s" : ""} for ${result.target} (${result.gene} · ${result.uniprot_id}) from ${result.total_sampled} sampled molecules.`}
          />

          {/* Stats strip */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Target", value: result.gene },
              { label: "Sampled", value: result.total_sampled },
              { label: "Returned", value: result.total_returned },
            ].map(({ label, value }) => (
              <div
                key={label}
                className="flex flex-col items-center py-3 rounded-xl"
                style={{ background: "var(--card-bg)", border: "1px solid var(--border)" }}
              >
                <span
                  className="text-[9px] font-semibold uppercase tracking-widest mb-1"
                  style={{ color: "var(--text-light)" }}
                >
                  {label}
                </span>
                <span
                  className="text-xl font-bold"
                  style={{ color: "var(--text-dark)" }}
                >
                  {value}
                </span>
              </div>
            ))}
          </div>

          {/* Candidate cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {result.candidates.map((c: any, i: number) => (
              <CandidateCard key={i} c={c} index={i} />
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