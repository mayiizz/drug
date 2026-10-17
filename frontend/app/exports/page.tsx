"use client";

import React, { useState } from "react";
import Card from "@/components/Card";
import { exportCSV, exportSDF } from "@/lib/api";
import { Download, FileText, Blocks, Loader2, CheckCircle } from "lucide-react";

export default function ExportsPage() {
  const [exporting, setExporting] = useState<string | null>(null);
  const [notif, setNotif] = useState(false);

  const handleExport = async (type: "csv" | "sdf") => {
    setExporting(type);
    try {
      const data = type === "csv" ? await exportCSV() : await exportSDF();
      const url = window.URL.createObjectURL(new Blob([data]));
      const a = document.createElement("a");
      a.href = url; a.setAttribute("download", `cura_candidates.${type}`);
      document.body.appendChild(a); a.click();
      setNotif(true); setTimeout(() => setNotif(false), 3000);
    } catch { setNotif(true); setTimeout(() => setNotif(false), 3000); }
    finally { setExporting(null); }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-16">
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}>// Exports</p>
        <h1 className="text-4xl font-bold mb-2" style={{ color: "var(--text-dark)", letterSpacing: "-1px" }}>Molecular Exports</h1>
        <p className="text-sm" style={{ color: "var(--text-light)", fontWeight: 300 }}>Download generated candidates and analysis results for lab research and modelling.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="flex flex-col items-center text-center p-10">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6"
            style={{ background: "rgba(63,184,232,0.1)", border: "1px solid rgba(63,184,232,0.2)" }}
          >
            <FileText className="w-8 h-8" style={{ color: "var(--blue-bright)" }} />
          </div>
          <h3 className="text-xl font-bold mb-2" style={{ color: "var(--text-dark)" }}>Export as CSV</h3>
          <p className="text-sm mb-8 leading-relaxed" style={{ color: "var(--text-light)", fontWeight: 300 }}>
            Best for spreadsheet analysis, R, or Python data processing. Includes all molecular properties and scores.
          </p>
          <button
            onClick={() => handleExport("csv")}
            disabled={!!exporting}
            className="w-full py-3.5 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-all hover:-translate-y-0.5 disabled:opacity-50"
            style={{ background: "var(--blue-bright)", color: "#fff", boxShadow: "0 4px 14px rgba(63,184,232,0.3)" }}
          >
            {exporting === "csv" ? <Loader2 className="animate-spin" size={18} /> : <Download size={18} />}
            {exporting === "csv" ? "Preparing…" : "Download CSV"}
          </button>
        </Card>

        <Card className="flex flex-col items-center text-center p-10">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6"
            style={{ background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)" }}
          >
            <Blocks className="w-8 h-8" style={{ color: "#10b981" }} />
          </div>
          <h3 className="text-xl font-bold mb-2" style={{ color: "var(--text-dark)" }}>Export as SDF</h3>
          <p className="text-sm mb-8 leading-relaxed" style={{ color: "var(--text-light)", fontWeight: 300 }}>
            Best for 3D modelling in PyMOL, ChimeraX, or Schrodinger Maestro.
          </p>
          <button
            onClick={() => handleExport("sdf")}
            disabled={!!exporting}
            className="w-full py-3.5 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-all hover:-translate-y-0.5 disabled:opacity-50"
            style={{ background: "#10b981", color: "#fff", boxShadow: "0 4px 14px rgba(16,185,129,0.25)" }}
          >
            {exporting === "sdf" ? <Loader2 className="animate-spin" size={18} /> : <Download size={18} />}
            {exporting === "sdf" ? "Preparing 3D Data…" : "Download SDF"}
          </button>
        </Card>
      </div>

      {notif && (
        <div
          className="fixed bottom-8 right-8 px-6 py-4 rounded-2xl flex items-center gap-3 shadow-2xl animate-fade-up"
          style={{ background: "#10b981", color: "#fff" }}
        >
          <CheckCircle className="w-5 h-5" />
          <div>
            <p className="font-semibold text-sm">Download Started</p>
            <p className="text-xs opacity-80">Your research data is being exported.</p>
          </div>
        </div>
      )}
    </div>
  );
}
