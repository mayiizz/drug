"use client";

import React, { useState, useEffect } from "react";
import Card from "@/components/Card";
import { analyzeMolecule, MoleculeAnalysisResponse } from "@/lib/api";
import { Beaker, Activity, Zap, Shield, Play, Loader2, CheckCircle2, RotateCw, AlertCircle, Info, Microscope } from "lucide-react";
import { useSearchParams } from "next/navigation";

const PropCard = ({ label, value, color }: { label: string; value: string; color: string }) => (
    <div className="p-4 rounded-xl flex items-center gap-4" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
        <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: `${color}18` }}>
            <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
        </div>
        <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>{label}</p>
            <p className="text-lg font-bold" style={{ color: "var(--text-dark)" }}>{value}</p>
        </div>
    </div>
);

const AdmetBadge = ({ label, value, goodValue }: { label: string; value: number | undefined; goodValue: number }) => {
    if (value === undefined) return (
        <div className="flex items-center justify-between p-3 rounded-xl" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
            <span className="text-sm" style={{ color: "var(--text-light)" }}>{label}</span>
            <span className="text-xs font-mono" style={{ color: "var(--text-light)" }}>—</span>
        </div>
    );
    const isGood = value === goodValue;
    const configs: Record<string, { pass: string; fail: string }> = {
        Toxicity: { pass: "Low Toxicity", fail: "High Toxicity" },
        Solubility: { pass: "Good Solubility", fail: "Poor Solubility" },
        Absorption: { pass: "High Absorption", fail: "Low Absorption" },
        Metabolism: { pass: "Stable", fail: "Unstable" },
    };
    const text = isGood ? configs[label]?.pass : configs[label]?.fail;
    return (
        <div className="flex items-center justify-between p-3 rounded-xl" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
            <span className="text-sm font-medium" style={{ color: "var(--text-mid)" }}>{label}</span>
            <span
                className="px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider"
                style={isGood
                    ? { background: "rgba(16,185,129,0.1)", color: "#059669", border: "1px solid rgba(16,185,129,0.2)" }
                    : { background: "rgba(239,68,68,0.1)", color: "#dc2626", border: "1px solid rgba(239,68,68,0.2)" }
                }
            >
                {text}
            </span>
        </div>
    );
};

export default function AnalysisPage() {
    const searchParams = useSearchParams();
    const querySmiles = searchParams.get("smiles");
    const [smiles, setSmiles] = useState("CC(=O)OC1=CC=CC=C1C(=O)O");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [results, setResults] = useState<MoleculeAnalysisResponse | null>(null);

    const handleAnalyze = async (inputSmiles?: string) => {
        const target = inputSmiles ?? smiles;
        setLoading(true);
        setError(null);
        try {
            const data = await analyzeMolecule(target);
            if (data.error) { setError(data.error); setResults(null); }
            else setResults(data);
        } catch (err: any) {
            setError(err.message || "Failed to analyze molecule");
            setResults(null);
        } finally { setLoading(false); }
    };

    useEffect(() => {
        if (querySmiles) { setSmiles(querySmiles); handleAnalyze(querySmiles); }
    }, [querySmiles]);

    return (
        <div className="max-w-6xl mx-auto space-y-8 pb-16">
            {/* Header */}
            <div>
                <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}>// Analysis</p>
                <h1 className="text-4xl font-bold mb-2" style={{ color: "var(--text-dark)", letterSpacing: "-1px" }}>Molecule Analysis</h1>
                <p className="text-sm" style={{ color: "var(--text-light)", fontWeight: 300 }}>Computational characterization of molecular properties and ADMET profiles from SMILES.</p>
            </div>

            {/* Input */}
            <Card title="Input SMILES" subtitle="Paste any valid SMILES notation">
                <div className="flex gap-3">
                    <input
                        type="text"
                        value={smiles}
                        onChange={(e) => setSmiles(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
                        placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O"
                        className="flex-1 rounded-xl px-4 py-3 text-sm outline-none transition-all"
                        style={{
                            background: "var(--bg)", border: "1px solid var(--border)",
                            color: "var(--text-dark)", fontFamily: "var(--font-space-mono)", fontSize: "12px",
                        }}
                    />
                    <button
                        onClick={() => handleAnalyze()}
                        disabled={loading}
                        className="px-7 py-3 rounded-xl text-sm font-semibold flex items-center gap-2 transition-all hover:-translate-y-0.5 disabled:opacity-50"
                        style={{ background: "var(--blue-bright)", color: "#fff", boxShadow: "0 4px 14px rgba(63,184,232,0.3)" }}
                    >
                        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                        {loading ? "Analyzing..." : "Analyze"}
                    </button>
                </div>
            </Card>

            {error && (
                <div className="p-4 rounded-xl flex items-start gap-3 animate-fade-up" style={{ background: "rgba(239,68,68,0.05)", border: "1px solid rgba(239,68,68,0.2)", color: "#ef4444" }}>
                    <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                    <p className="text-sm">{error}</p>
                </div>
            )}

            {results && (
                <div className="space-y-6 animate-fade-up">
                    {/* Top stats */}
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                        <PropCard label="Mol. Weight" value={`${results.molecular_weight?.toFixed(1) ?? "—"} g/mol`} color="#3fb8e8" />
                        <PropCard label="LogP" value={results.logp?.toFixed(2) ?? "—"} color="#10b981" />
                        <PropCard label="TPSA" value={`${results.tpsa?.toFixed(1) ?? "—"} Å²`} color="#8b5cf6" />
                        <PropCard label="Drug Likeness" value={`${((results.drug_likeness ?? 0) * 100).toFixed(0)}%`} color="#2196c9" />
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <div className="lg:col-span-2 space-y-6">
                            {/* Descriptors */}
                            <Card title="Molecular Descriptors" subtitle="Fundamental physicochemical properties">
                                <div className="grid grid-cols-2 gap-3 mt-1">
                                    <PropCard label="H-Bond Donors" value={String(results.h_donors ?? "—")} color="#6366f1" />
                                    <PropCard label="H-Bond Acceptors" value={String(results.h_acceptors ?? "—")} color="#f59e0b" />
                                    <PropCard label="Rotatable Bonds" value={String(results.rotatable_bonds ?? "—")} color="#f97316" />
                                    <PropCard
                                        label="Lipinski Violations"
                                        value={String(results.lipinski_violations ?? "—")}
                                        color={results.lipinski_violations === 0 ? "#10b981" : "#ef4444"}
                                    />
                                </div>
                            </Card>

                            {/* Structure */}
                            <Card>
                                <div className="flex items-center gap-3 mb-5">
                                    <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: "rgba(63,184,232,0.1)" }}>
                                        <Microscope className="w-5 h-5" style={{ color: "var(--blue-bright)" }} />
                                    </div>
                                    <div>
                                        <h4 className="font-semibold text-sm" style={{ color: "var(--text-dark)" }}>Molecule Structure</h4>
                                        <p className="text-xs" style={{ color: "var(--text-light)" }}>2D representation from SMILES</p>
                                    </div>
                                </div>
                                <div
                                    className="rounded-xl p-6 flex justify-center"
                                    style={{ background: "var(--bg)", border: "1px solid var(--border)" }}
                                >
                                    {results.image ? (
                                        <img src={results.image} alt="Molecule Structure" className="max-h-60 rounded-lg" />
                                    ) : (
                                        <div className="h-48 flex flex-col items-center justify-center gap-2">
                                            <div className="w-8 h-8 rounded-full border-2 animate-spin" style={{ borderColor: "var(--border)", borderTopColor: "var(--blue-bright)" }} />
                                            <span className="text-xs" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>Generating...</span>
                                        </div>
                                    )}
                                </div>
                            </Card>
                        </div>

                        {/* ADMET + Lipinski */}
                        <div className="space-y-6">
                            <Card title="ADMET Heuristics" subtitle="Physicochemical screening">
                                <div className="space-y-2.5 mt-1">
                                    <AdmetBadge label="Toxicity" value={results.toxicity} goodValue={0} />
                                    <AdmetBadge label="Solubility" value={results.solubility} goodValue={1} />
                                    <AdmetBadge label="Absorption" value={results.absorption} goodValue={1} />
                                    <AdmetBadge label="Metabolism" value={results.metabolism} goodValue={1} />
                                </div>
                                <div
                                    className="mt-4 flex items-start gap-2 p-3 rounded-xl"
                                    style={{ background: "rgba(63,184,232,0.05)", border: "1px solid rgba(63,184,232,0.15)" }}
                                >
                                    <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" style={{ color: "var(--blue-mid)" }} />
                                    <p className="text-[11px] leading-relaxed" style={{ color: "var(--text-light)" }}>
                                        Fast heuristic screens based on physicochemical rules. Results are indicative — not a replacement for full ADMET modelling.
                                    </p>
                                </div>
                            </Card>

                            <Card title="Lipinski Rule of Five">
                                <div
                                    className="p-5 rounded-xl flex flex-col items-center text-center gap-2 mt-1"
                                    style={
                                        (results.lipinski_violations ?? 0) === 0
                                            ? { background: "rgba(16,185,129,0.07)", border: "1px solid rgba(16,185,129,0.2)" }
                                            : { background: "rgba(239,68,68,0.07)", border: "1px solid rgba(239,68,68,0.2)" }
                                    }
                                >
                                    <p className="text-[10px] uppercase font-bold tracking-widest" style={{ color: "var(--text-light)" }}>Status</p>
                                    <p className="text-2xl font-bold" style={{ color: (results.lipinski_violations ?? 0) === 0 ? "#059669" : "#dc2626" }}>
                                        {(results.lipinski_violations ?? 0) === 0 ? "PASSED" : "FAILED"}
                                    </p>
                                    <p className="text-xs" style={{ color: "var(--text-light)" }}>
                                        {results.lipinski_violations ?? 0} violation{(results.lipinski_violations ?? 0) !== 1 ? "s" : ""} detected
                                    </p>
                                </div>
                            </Card>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
