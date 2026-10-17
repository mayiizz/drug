"use client";

import React, { useState } from "react";
import Card from "@/components/Card";
import ResearchBriefPanel from "@/components/ResearchBriefPanel";
import { BookOpen, Loader2, Sparkles } from "lucide-react";

const RESEARCH_INPUT_HINT =
  "Please provide a valid molecule or use a natural language query.";

function isPlaceholderSmiles(s: string): boolean {
  const t = s.trim();
  if (!t) return true;
  const low = t.toLowerCase().replace(/\.+$/u, "");
  if (low === "smiles string" || low === "smiles string…") return true;
  if (low.startsWith("smiles string")) return true;
  return false;
}

export default function ResearchPage() {
  const [query, setQuery] = useState("");
  const [smiles, setSmiles] = useState("");
  const [target, setTarget] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [chunks, setChunks] = useState<string[]>([]);
  const [clientError, setClientError] = useState("");

  const canSubmit =
    query.trim().length > 0 ||
    (!isPlaceholderSmiles(smiles) && smiles.trim().length > 0);

  const runResearch = async () => {
    setClientError("");
    if (!canSubmit) {
      setClientError(RESEARCH_INPUT_HINT);
      return;
    }
    setChunks([]);
    setStreaming(true);
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";
      const res = await fetch(`${apiBase}/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          smiles: smiles.trim(),
          target: target.trim(),
        }),
      });
      if (!res.ok) {
        const errText = await res.text();
        let detail = RESEARCH_INPUT_HINT;
        try {
          const j = JSON.parse(errText);
          detail = j.detail ?? (Array.isArray(j.detail) ? j.detail[0]?.msg : detail);
        } catch {
          if (res.status === 422) detail = RESEARCH_INPUT_HINT;
        }
        setClientError(typeof detail === "string" ? detail : RESEARCH_INPUT_HINT);
        setStreaming(false);
        return;
      }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value);
        const lines = text.split("\n").filter((l) => l.startsWith("data:"));
        for (const line of lines) {
          const data = line.replace("data: ", "").trim();
          if (data === "[DONE]") {
            setStreaming(false);
            break;
          }
          try {
            const parsed = JSON.parse(data);
            if (parsed.content) setChunks((prev) => [...prev, parsed.content]);
          } catch {
            /* ignore */
          }
        }
      }
    } finally {
      setStreaming(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-16">
      <div className="flex items-center gap-4">
        <div
          className="w-11 h-11 rounded-xl flex items-center justify-center"
          style={{
            background: "rgba(63,184,232,0.1)",
            border: "1px solid rgba(63,184,232,0.2)",
          }}
        >
          <BookOpen className="w-6 h-6" style={{ color: "var(--blue-mid)" }} />
        </div>
        <div>
          <p
            className="text-[10px] font-semibold uppercase tracking-widest mb-1"
            style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)" }}
          >
            // Research Tools
          </p>
          <h1
            className="text-4xl font-bold"
            style={{ color: "var(--text-dark)", letterSpacing: "-1px" }}
          >
            Literature Research Agent
          </h1>
        </div>
      </div>

      <Card
        title="Research Query"
        subtitle="Natural language question — or SMILES + target (e.g. aspirin + COX-2). PubChem resolves names from SMILES."
      >
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium mb-1.5 opacity-80">
              Mode 1 — Natural language
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder='e.g. "What is the role of COX-2 in inflammation?"'
              rows={3}
              className="w-full px-4 py-3 rounded-xl text-sm outline-none resize-y min-h-[88px]"
              style={{
                background: "var(--bg)",
                border: "1px solid var(--border)",
                color: "var(--text-dark)",
              }}
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className="block text-xs font-medium mb-1.5 opacity-80">
                Mode 2 — SMILES (optional if you use the question above)
              </label>
              <input
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="SMILES string…"
                className="w-full px-4 py-3 rounded-xl text-xs outline-none"
                style={{
                  background: "var(--bg)",
                  border: "1px solid var(--border)",
                  color: "var(--text-dark)",
                  fontFamily: "var(--font-space-mono)",
                }}
              />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5 opacity-80">
                Target (e.g. COX-2, EGFR)
              </label>
              <input
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="Target (e.g. COX-2)…"
                className="w-full px-4 py-3 rounded-xl text-sm outline-none"
                style={{
                  background: "var(--bg)",
                  border: "1px solid var(--border)",
                  color: "var(--text-dark)",
                }}
              />
            </div>
          </div>
          {clientError ? (
            <p className="text-sm text-amber-600 dark:text-amber-400">{clientError}</p>
          ) : null}
          <button
            type="button"
            onClick={runResearch}
            disabled={streaming || !canSubmit}
            className="px-7 py-3 rounded-xl text-sm font-semibold flex items-center gap-2 transition-all hover:-translate-y-0.5 disabled:opacity-50 shrink-0"
            style={{
              background: "var(--blue-bright)",
              color: "#fff",
              boxShadow: "0 4px 14px rgba(63,184,232,0.3)",
            }}
          >
            {streaming ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            {streaming ? "Researching…" : "Research"}
          </button>
        </div>
      </Card>

      <ResearchBriefPanel chunks={chunks} streaming={streaming} />
    </div>
  );
}
