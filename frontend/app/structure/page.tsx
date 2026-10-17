"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
import Card from "@/components/Card";
import {
  Atom, Loader2, AlertCircle, CheckCircle2, Zap, Info, Download, Box,
} from "lucide-react";
import { loadAlphaFoldStructure, type StructureJob } from "@/lib/alphafold";

const Protein3DViewer = dynamic(() => import("@/components/Protein3DViewer"), {
  ssr: false,
  loading: () => (
    <div
      className="flex items-center justify-center rounded-2xl gap-2 text-sm"
      style={{ height: 420, background: "#0b1824", color: "#7dd9f7", border: "1px solid var(--border)" }}
    >
      <Loader2 size={16} className="animate-spin" />
      Loading viewer…
    </div>
  ),
});

const EXAMPLES = [
  { id: "P00533", label: "EGFR" },
  { id: "P00520", label: "ABL1" },
  { id: "P04637", label: "TP53" },
  { id: "P69905", label: "HBA" },
  { id: "P0DTD1", label: "SARS-CoV-2" },
];

export default function StructurePage() {
  const [query, setQuery] = useState("P00533");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<StructureJob | null>(null);

  const handleSubmit = async (override?: string) => {
    const q = (override ?? query).trim();
    if (!q) {
      setError("Enter a UniProt accession or gene name (e.g. P00533 or EGFR).");
      return;
    }

    setQuery(q);
    setLoading(true);
    setError(null);
    setJob(null);

    try {
      const result = await loadAlphaFoldStructure(q);
      setJob(result);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to load structure.";
      if (msg.includes("404") || msg.toLowerCase().includes("not found")) {
        setError(
          `No AlphaFold structure found for "${q}". Try a UniProt accession with a public AlphaFold model (e.g. P00533 for EGFR).`
        );
      } else if (msg.includes("Failed to fetch") || msg.includes("NetworkError") || msg.includes("CORS")) {
        setError("Could not reach AlphaFold DB. Check your network connection and try again.");
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!job) return;
    const blob = new Blob([job.pdbText], { type: "chemical/x-pdb" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `AF-${job.uniprotId}.pdb`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-16">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div
          className="w-11 h-11 rounded-xl flex items-center justify-center"
          style={{ background: "rgba(63,184,232,0.1)", border: "1px solid rgba(63,184,232,0.2)" }}
        >
          <Atom className="w-6 h-6" style={{ color: "var(--blue-mid)" }} />
        </div>
        <div>
          <p
            className="text-[10px] font-semibold uppercase tracking-widest mb-1"
            style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}
          >
            AlphaFold Structure Viewer
          </p>
          <h1
            className="text-4xl font-bold"
            style={{ color: "var(--text-dark)", letterSpacing: "-1px" }}
          >
            3D Protein Structure
          </h1>
        </div>
      </div>

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
          Enter a UniProt accession or gene name to load the predicted 3D structure from the{" "}
          <strong>AlphaFold Protein Structure Database</strong> (DeepMind / EMBL-EBI).
          Rotate with the mouse, scroll to zoom, and switch between secondary structure, surface, and confidence views.
        </span>
      </div>

      <Card
        title="Predict Structure"
        subtitle="Fetch and visualize AlphaFold models by UniProt ID or gene symbol"
      >
        <div className="space-y-4 mt-1">
          <div>
            <label
              className="text-[10px] font-semibold uppercase tracking-widest block mb-2"
              style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}
            >
              UniProt Identifier / Gene
            </label>
            <input
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setError(null);
              }}
              onKeyDown={(e) => e.key === "Enter" && !loading && handleSubmit()}
              placeholder="e.g. P00533, EGFR, BRAF"
              disabled={loading}
              className="w-full px-4 py-3 rounded-xl text-sm font-semibold outline-none"
              style={{
                background: "var(--bg)",
                border: "1px solid var(--border)",
                color: "var(--text-dark)",
                opacity: loading ? 0.6 : 1,
              }}
            />
            <div className="flex flex-wrap items-center gap-2 mt-3">
              <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: "var(--text-light)" }}>
                Examples
              </span>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.id}
                  type="button"
                  disabled={loading}
                  onClick={() => handleSubmit(ex.id)}
                  className="px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all hover:-translate-y-0.5"
                  style={{
                    background: "rgba(63,184,232,0.08)",
                    border: "1px solid rgba(63,184,232,0.2)",
                    color: "var(--blue-mid)",
                  }}
                >
                  {ex.id}
                  <span style={{ color: "var(--text-light)", fontWeight: 500 }}> · {ex.label}</span>
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={() => handleSubmit()}
            disabled={loading}
            className="w-full py-3.5 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-all hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
            style={{
              background: "linear-gradient(135deg, var(--blue-bright), var(--blue-deep))",
              color: "#fff",
              boxShadow: loading ? "none" : "0 4px 14px rgba(63,184,232,0.3)",
            }}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading AlphaFold model…
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                Submit Prediction
              </>
            )}
          </button>
        </div>
      </Card>

      {error && !loading && (
        <div
          className="flex items-start gap-3 p-4 rounded-xl text-sm"
          style={{
            background: "rgba(239,68,68,0.07)",
            border: "1px solid rgba(239,68,68,0.2)",
            color: "#dc2626",
          }}
        >
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {job && !loading && (
        <div className="space-y-5">
          <Card title="Prediction Status" subtitle="AlphaFold DB model ready for interactive viewing">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-1">
              <div className="px-4 py-3 rounded-xl" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                <p className="text-[9px] font-semibold uppercase tracking-widest mb-1" style={{ color: "var(--text-light)" }}>
                  UniProt ID
                </p>
                <p className="text-sm font-bold" style={{ color: "var(--text-dark)" }}>
                  {job.uniprotId}
                  {job.gene ? (
                    <span className="font-medium ml-2" style={{ color: "var(--text-light)" }}>
                      ({job.gene})
                    </span>
                  ) : null}
                </p>
              </div>
              <div className="px-4 py-3 rounded-xl" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                <p className="text-[9px] font-semibold uppercase tracking-widest mb-1" style={{ color: "var(--text-light)" }}>
                  Status
                </p>
                <p className="text-sm font-bold flex items-center gap-1.5" style={{ color: "#059669" }}>
                  <CheckCircle2 size={15} />
                  {job.status}
                </p>
              </div>
              <div className="px-4 py-3 rounded-xl" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                <p className="text-[9px] font-semibold uppercase tracking-widest mb-1" style={{ color: "var(--text-light)" }}>
                  Submitted
                </p>
                <p className="text-sm font-bold" style={{ color: "var(--text-dark)" }}>
                  {job.submittedAt}
                </p>
              </div>
            </div>
          </Card>

          <Card
            title="3D Visualizer"
            subtitle="Drag to rotate · scroll to zoom · right-drag to pan"
          >
            <Protein3DViewer
              pdbText={job.pdbText}
              uniprotId={job.uniprotId}
              activeSiteResidues={job.activeSiteResidues}
            />

            <div className="flex justify-end gap-3 mt-4 pt-4" style={{ borderTop: "1px solid var(--border)" }}>
              <a
                href={`https://alphafold.ebi.ac.uk/entry/${job.uniprotId}`}
                target="_blank"
                rel="noreferrer"
                className="px-4 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all hover:-translate-y-0.5"
                style={{
                  background: "transparent",
                  border: "1px solid var(--border)",
                  color: "var(--text-dark)",
                }}
              >
                <Box size={14} />
                Open in AlphaFold DB
              </a>
              <button
                type="button"
                onClick={handleDownload}
                className="px-4 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all hover:-translate-y-0.5"
                style={{
                  background: "transparent",
                  border: "1px solid var(--border)",
                  color: "var(--text-dark)",
                }}
              >
                <Download size={14} />
                Download PDB
              </button>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
