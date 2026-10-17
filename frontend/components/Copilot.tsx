"use client";

import React, { useState, useEffect, useRef } from "react";
import { Bot, X, Send, Sparkles, Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { parseCopilotIntent } from "@/lib/ai";

export default function Copilot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "👋 Hi! I'm **CURA Copilot**, your AI drug discovery partner.\n\nTry asking:\n• *'Find a candidate for EGFR lung cancer'*\n• *'Navigate to the neuro-network tool'*\n• *'What makes a good drug candidate?'*",
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => { setIsMounted(true); }, []);
  useEffect(() => { scrollRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, isTyping]);

  const handleAction = async (msg: string) => {
    if (!msg.trim()) return;
    setMessages((prev) => [...prev, { role: "user", content: msg }]);
    setInput("");
    setIsTyping(true);
    try {
      const data = await parseCopilotIntent(msg, messages);
      let responseText = data.direct_response || data.raw_intent_summary || "Processing…";
      if (data.intent === "generate") {
        const q = new URLSearchParams();
        if (data.target) q.set("target", data.target);
        q.set("trigger", "true");
        router.push(`/discovery?${q.toString()}`);
        if (!data.direct_response) responseText = `🔬 Navigating to discovery for **${data.target || "target"}**…`;
      } else if (data.intent === "navigate") {
        const pages: Record<string, string> = { discovery: "/discovery", analysis: "/analysis", network: "/network", similar: "/similar-drugs", safety: "/side-effects" };
        router.push(pages[data.page] || "/");
        if (!data.direct_response) responseText = `🚀 Navigating to **${data.page}**…`;
      } else if (data.intent === "analyze") {
        router.push("/analysis");
        if (!data.direct_response) responseText = `🧪 Taking you to the analysis lab.`;
      }
      setMessages((prev) => [...prev, { role: "assistant", content: responseText }]);
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: "Error processing request. Please try again." }]);
    } finally { setIsTyping(false); }
  };

  if (!isMounted) return null;

  return (
    <div className="fixed bottom-6 right-6 z-[100]">
      {/* FAB */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-13 h-13 rounded-full flex items-center justify-center text-white shadow-2xl transition-all duration-300 hover:scale-110 active:scale-95 relative"
        style={{
          width: 52, height: 52,
          background: isOpen
            ? "var(--text-mid)"
            : "linear-gradient(135deg, var(--blue-bright), var(--blue-deep))",
          boxShadow: "0 8px 28px rgba(63,184,232,0.35)",
        }}
      >
        {isOpen ? <X size={22} /> : <Bot size={24} />}
        {!isOpen && <span className="absolute -top-0.5 -right-0.5 w-3 h-3 bg-emerald-500 border-2 border-white rounded-full animate-pulse" />}
      </button>

      {/* Drawer */}
      <div
        className="absolute bottom-16 right-0 flex flex-col overflow-hidden"
        style={{
          width: 400, height: 540,
          background: "#fff",
          border: "1px solid var(--border)",
          borderRadius: 20,
          boxShadow: "0 20px 60px rgba(15,31,46,0.15)",
          transformOrigin: "bottom right",
          transition: "all 0.3s cubic-bezier(0.34,1.56,0.64,1)",
          transform: isOpen ? "scale(1) translateY(0)" : "scale(0.92) translateY(10px)",
          opacity: isOpen ? 1 : 0,
          pointerEvents: isOpen ? "all" : "none",
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: "rgba(63,184,232,0.1)" }}>
              <Bot size={18} style={{ color: "var(--blue-mid)" }} />
            </div>
            <div>
              <h3 className="text-sm font-bold flex items-center gap-1.5" style={{ color: "var(--text-dark)" }}>
                CURA Copilot <Sparkles size={11} style={{ color: "var(--blue-bright)" }} />
              </h3>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>Llama-3 Powered</span>
              </div>
            </div>
          </div>
          <button onClick={() => setIsOpen(false)} style={{ color: "var(--text-light)" }}>
            <X size={18} />
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              {m.role === "assistant" && (
                <div className="w-7 h-7 rounded-lg flex items-center justify-center mr-2 mt-1 shrink-0" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                  <Bot size={13} style={{ color: "var(--blue-mid)" }} />
                </div>
              )}
              <div
                className="max-w-[80%] p-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap"
                style={
                  m.role === "user"
                    ? { background: "var(--blue-bright)", color: "#fff", borderTopRightRadius: 4 }
                    : { background: "var(--bg)", color: "var(--text-mid)", border: "1px solid var(--border)", borderTopLeftRadius: 4 }
                }
              >
                {m.content.split("**").map((part, j) =>
                  j % 2 === 1
                    ? <b key={j} style={{ color: m.role === "user" ? "#fff" : "var(--text-dark)" }}>{part}</b>
                    : part
                )}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                <Bot size={13} style={{ color: "var(--blue-mid)" }} />
              </div>
              <div className="px-4 py-3 rounded-2xl flex gap-1" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
                {[0, 0.15, 0.3].map((d, i) => (
                  <div key={i} className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: "var(--blue-bright)", animationDelay: `${d}s` }} />
                ))}
              </div>
            </div>
          )}
          <div ref={scrollRef} />
        </div>

        {/* Input */}
        <div className="p-4" style={{ borderTop: "1px solid var(--border)" }}>
          <div className="relative">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAction(input)}
              placeholder="Ask me to find drug candidates…"
              className="w-full pr-12 pl-4 py-3 rounded-xl text-sm outline-none"
              style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--text-dark)" }}
            />
            <button
              disabled={!input.trim() || isTyping}
              onClick={() => handleAction(input)}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg flex items-center justify-center transition-all disabled:opacity-40"
              style={{ background: "var(--blue-bright)", color: "#fff" }}
            >
              <Send size={15} />
            </button>
          </div>
          <p className="text-[9px] text-center mt-2.5 font-semibold uppercase tracking-widest" style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}>
            CURA Intelligence · v2.0
          </p>
        </div>
      </div>
    </div>
  );
}
