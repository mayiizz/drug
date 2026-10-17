"use client";

import React, { useEffect, useRef } from "react";
import { BookOpen, Loader2, AlertCircle } from "lucide-react";

interface Props { chunks: string[]; streaming: boolean; error?: string; }

function SimpleMarkdown({ text }: { text: string }) {
  return (
    <div className="space-y-2 text-sm leading-relaxed" style={{ color: "var(--text-mid)" }}>
      {text.split("\n").map((line, i) => {
        if (line.startsWith("### ")) return <h3 key={i} className="text-base font-semibold mt-4 mb-1" style={{ color: "var(--text-dark)" }}>{line.slice(4)}</h3>;
        if (line.startsWith("## "))  return <h2 key={i} className="text-lg font-bold mt-6 mb-2 pb-1" style={{ color: "var(--blue-mid)", borderBottom: "1px solid var(--border)" }}>{line.slice(3)}</h2>;
        if (line.startsWith("# "))   return <h1 key={i} className="text-xl font-bold mt-6 mb-2" style={{ color: "var(--text-dark)" }}>{line.slice(2)}</h1>;
        if (line.startsWith("- ") || line.startsWith("* "))
          return <div key={i} className="flex gap-2"><span style={{ color: "var(--blue-bright)" }}>•</span><span style={{ fontWeight: 300 }}>{line.slice(2)}</span></div>;
        if (/^\d+\.\s/.test(line))
          return <div key={i} className="flex gap-2"><span style={{ color: "var(--blue-mid)", fontFamily: "var(--font-space-mono)", fontSize: 11 }}>{line.match(/^\d+/)![0]}.</span><span style={{ fontWeight: 300 }}>{line.replace(/^\d+\.\s/, "")}</span></div>;
        if (line.includes("**")) {
          const parts = line.split(/\*\*(.*?)\*\*/g);
          return <p key={i}>{parts.map((p, j) => j % 2 === 1 ? <strong key={j} style={{ color: "var(--text-dark)", fontWeight: 600 }}>{p}</strong> : <span key={j}>{p}</span>)}</p>;
        }
        if (!line.trim()) return <div key={i} className="h-2" />;
        return <p key={i} style={{ fontWeight: 300 }}>{line}</p>;
      })}
    </div>
  );
}

export default function ResearchBriefPanel({ chunks, streaming, error }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const fullText = chunks.join("");
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chunks]);

  return (
    <div className="rounded-2xl overflow-hidden" style={{ border: "1px solid var(--border)", background: "#fff" }}>
      <div className="flex items-center gap-3 px-5 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "rgba(63,184,232,0.1)", border: "1px solid rgba(63,184,232,0.2)" }}>
          <BookOpen className="w-4 h-4" style={{ color: "var(--blue-mid)" }} />
        </div>
        <span className="text-sm font-semibold" style={{ color: "var(--text-dark)" }}>Research Brief</span>
        {streaming && (
          <span className="ml-auto flex items-center gap-2 text-xs" style={{ color: "var(--blue-mid)" }}>
            <Loader2 className="w-3 h-3 animate-spin" /> Agent researching…
          </span>
        )}
        {!streaming && fullText && (
          <span className="ml-auto text-xs font-semibold" style={{ color: "#059669" }}>Complete</span>
        )}
      </div>

      <div className="p-5 max-h-[520px] overflow-y-auto">
        {error && (
          <div className="flex items-start gap-3 p-4 rounded-xl text-sm" style={{ background: "rgba(239,68,68,0.07)", border: "1px solid rgba(239,68,68,0.2)", color: "#dc2626" }}>
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" /><span>{error}</span>
          </div>
        )}
        {!fullText && !streaming && !error && (
          <p className="text-sm text-center py-12" style={{ color: "var(--text-light)" }}>
            Enter a SMILES and target, then click <span style={{ color: "var(--blue-mid)" }}>Research</span>.
          </p>
        )}
        {streaming && !fullText && (
          <div className="space-y-3 animate-pulse">
            {[80, 60, 90, 50, 70].map((w, i) => (
              <div key={i} className="h-3 rounded" style={{ width: `${w}%`, background: "var(--bg)" }} />
            ))}
          </div>
        )}
        {fullText && <SimpleMarkdown text={fullText} />}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
