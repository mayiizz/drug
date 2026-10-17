"use client";

import React from "react";
import Link from "next/link";
import { FlaskConical, Search, ShieldCheck, Network, Sparkles, Dna, ArrowRight, Zap, TrendingUp } from "lucide-react";
import Card from "@/components/Card";

const tools = [
  { title: "Molecule Analysis", subtitle: "Complete property profiling — MW, LogP, TPSA, QED, ADMET.", href: "/analysis", icon: FlaskConical, accent: "#3fb8e8" },
  { title: "Similar Drug Finder", subtitle: "Find structural analogs using fingerprint similarity.", href: "/similar-drugs", icon: Search, accent: "#2196c9" },
  { title: "Side Effects", subtitle: "Predict adverse drug reactions from SIDER data.", href: "/side-effects", icon: ShieldCheck, accent: "#f59e0b" },
  { title: "Bio Network", subtitle: "Drug–protein–disease interaction graph.", href: "/network", icon: Network, accent: "#10b981" },
];

const pipeline = [
  { label: "SMILES Validation", status: "Ready", pct: 100 },
  { label: "Fingerprint Engine", status: "Optimized", pct: 100 },
  { label: "Protein Binding", status: "Simulating", pct: 65 },
  { label: "Multi-Objective Rank", status: "Active", pct: 90 },
];

export default function Dashboard() {
  return (
    <div className="max-w-7xl mx-auto space-y-10 pb-20">

      {/* Hero */}
      <div className="pt-6 pb-2">
        <div
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold mb-5"
          style={{ background: "rgba(63,184,232,0.1)", border: "1px solid rgba(63,184,232,0.25)", color: "var(--blue-mid)" }}
        >
          <Zap size={12} className="animate-pulse-dot" />
          CURA  AI Engine Active
        </div>
        <h1
          className="text-5xl font-extrabold tracking-tight leading-tight mb-4"
          style={{ color: "var(--text-dark)", letterSpacing: "-1.5px" }}
        >
          Modern AI<br />
          <span style={{ color: "var(--blue-mid)" }}>Drug Generation</span>
        </h1>
        <p className="text-base max-w-xl" style={{ color: "var(--text-light)", fontWeight: 300, lineHeight: 1.75 }}>
          Accelerate pharmacological research with deep molecular embeddings,
          ADMET predictions, and generative lead optimization.
        </p>
      </div>

      {/* Tool cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {tools.map((t) => (
          <Link href={t.href} key={t.href}>
            <Card className="h-full group hover:-translate-y-1 cursor-pointer"
              style={{ transition: "all 0.22s", borderColor: "var(--border)" }}
            >
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center mb-5 transition-all group-hover:scale-110"
                style={{ background: `${t.accent}18` }}
              >
                <t.icon size={20} style={{ color: t.accent }} />
              </div>
              <h3 className="font-semibold text-sm mb-2" style={{ color: "var(--text-dark)" }}>{t.title}</h3>
              <p className="text-xs leading-relaxed mb-5" style={{ color: "var(--text-light)", fontWeight: 300 }}>{t.subtitle}</p>
              <div className="flex items-center gap-1.5 text-xs font-semibold" style={{ color: "var(--blue-mid)" }}>
                Explore <ArrowRight size={12} className="group-hover:translate-x-1 transition-transform" />
              </div>
            </Card>
          </Link>
        ))}
      </div>

      {/* Discovery spotlight + pipeline */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

        {/* Discovery CTA */}
        <div
          className="lg:col-span-2 rounded-2xl p-8"
          style={{
            background: "linear-gradient(135deg, #7ec8e3 0%, #5bbde0 45%, #3aa5cc 100%)",
            boxShadow: "0 12px 40px rgba(63,184,232,0.25)",
          }}
        >
          <div className="flex flex-col md:flex-row gap-8 items-center">
            <div className="flex-1 space-y-4">
              <p className="text-xs font-bold uppercase tracking-widest text-white/70"
                style={{ fontFamily: "var(--font-space-mono)" }}>
                // AI Candidate Discovery
              </p>
              <h2 className="text-2xl font-bold text-white" style={{ letterSpacing: "-0.5px" }}>
                MolGPT + DeepPurpose<br />Binding Validation
              </h2>
              <p className="text-sm text-white/75" style={{ fontWeight: 300, lineHeight: 1.7 }}>
                Every candidate is pre-checked against your protein target before generation.
                9-parameter scoring. Top 10 ranked results.
              </p>
              <div className="flex gap-3 pt-1">
                <Link
                  href="/discovery"
                  className="px-6 py-3 rounded-xl text-sm font-bold transition-all hover:-translate-y-0.5"
                  style={{ background: "var(--green-lime)", color: "var(--text-dark)", boxShadow: "0 4px 14px rgba(200,240,103,0.35)" }}
                >
                  Start Discovery →
                </Link>
                <Link
                  href="/protein"
                  className="px-6 py-3 rounded-xl text-sm font-semibold transition-all"
                  style={{ background: "rgba(255,255,255,0.2)", color: "#fff", border: "1px solid rgba(255,255,255,0.3)" }}
                >
                  Bind Target
                </Link>
              </div>
            </div>
            <div
              className="w-40 h-40 rounded-full flex items-center justify-center shrink-0"
              style={{ background: "rgba(255,255,255,0.15)", border: "2px solid rgba(255,255,255,0.2)" }}
            >
              <div
                className="w-28 h-28 rounded-full flex items-center justify-center animate-spin-slow"
                style={{ border: "2px dashed rgba(255,255,255,0.4)" }}
              >
                <TrendingUp className="text-white" size={32} />
              </div>
            </div>
          </div>
        </div>

        {/* Pipeline status */}
        <Card title="Pipeline Status" subtitle="Current engine readiness">
          <div className="space-y-5 mt-2">
            {pipeline.map((s, i) => (
              <div key={i}>
                <div className="flex justify-between text-xs font-medium mb-1.5">
                  <span style={{ color: "var(--text-mid)" }}>{s.label}</span>
                  <span style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)", fontSize: "10px" }}>{s.status}</span>
                </div>
                <div className="h-1.5 w-full rounded-full" style={{ background: "var(--bg)" }}>
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${s.pct}%`, background: s.pct === 100 ? "var(--blue-bright)" : "var(--blue-pale)" }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

    </div>
  );
}
