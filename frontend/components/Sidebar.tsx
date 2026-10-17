"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, Search, FlaskConical, AlertTriangle,
  Network, Dna, Sparkles, BookOpen, Atom,
} from "lucide-react";

const navItems = [
  {
    group: "Analysis",
    items: [
      { name: "Dashboard", href: "/", icon: LayoutDashboard },
      { name: "Molecule Analysis", href: "/analysis", icon: FlaskConical },
      { name: "3D Structure", href: "/structure", icon: Atom },
    ],
  },
  {
    group: "Drug Intelligence",
    items: [
      { name: "Similar Drugs", href: "/similar-drugs", icon: Search },
      { name: "Side Effects", href: "/side-effects", icon: AlertTriangle },
      { name: "Drug Network", href: "/network", icon: Network },
    ],
  },
  {
    group: "Research Tools",
    items: [
      { name: "Protein Target Drug Generation", href: "/protein", icon: Dna },
      { name: "Seed Candidate Drug Generation", href: "/discovery", icon: Sparkles },
      { name: "Research Agent", href: "/research", icon: BookOpen },
    ],
  },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside
      className="w-64 h-screen sticky top-0 flex flex-col z-50 shrink-0"
      style={{
        background: "#ffffff",
        borderRight: "1px solid var(--border)",
      }}
    >
      {/* Logo */}
      <div className="p-6 pb-4">
        <div className="flex items-center gap-3 mb-8">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: "var(--blue-bright)" }}
          >
            <svg viewBox="0 0 18 18" fill="none" stroke="white" strokeWidth="1.8" width="18" height="18">
              <circle cx="9" cy="9" r="2.5" />
              <circle cx="3" cy="4" r="1.3" /><circle cx="15" cy="4" r="1.3" />
              <circle cx="3" cy="14" r="1.3" /><circle cx="15" cy="14" r="1.3" />
              <line x1="9" y1="9" x2="3" y2="4" /><line x1="9" y1="9" x2="15" y2="4" />
              <line x1="9" y1="9" x2="3" y2="14" /><line x1="9" y1="9" x2="15" y2="14" />
            </svg>
          </div>
          <span className="text-xl font-bold tracking-tight" style={{ color: "var(--text-dark)", letterSpacing: "-0.3px" }}>
            CURA <span style={{ color: "var(--blue-mid)" }}>AI</span>
          </span>
        </div>

        {/* Nav */}
        <nav className="space-y-7">
          {navItems.map((group) => (
            <div key={group.group}>
              <p
                className="text-[10px] font-semibold uppercase tracking-widest mb-3 px-2"
                style={{ color: "var(--text-light)", fontFamily: "var(--font-space-mono)" }}
              >
                {group.group}
              </p>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const isActive = pathname === item.href;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className="flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all text-sm font-medium group"
                      style={{
                        background: isActive ? "rgba(63,184,232,0.1)" : "transparent",
                        color: isActive ? "var(--blue-mid)" : "var(--text-light)",
                        border: isActive ? "1px solid rgba(63,184,232,0.25)" : "1px solid transparent",
                        fontWeight: isActive ? 600 : 400,
                      }}
                    >
                      <item.icon
                        className="w-4 h-4 transition-colors"
                        style={{ color: isActive ? "var(--blue-mid)" : "var(--text-light)" }}
                      />
                      {item.name}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>
      </div>

      {/* Status footer */}
      <div className="mt-auto p-4" style={{ borderTop: "1px solid var(--border)" }}>
        <div className="p-3 rounded-xl" style={{ background: "var(--bg)", border: "1px solid var(--border)" }}>
          <p className="text-[10px] font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--text-light)" }}>Status</p>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs font-medium" style={{ color: "var(--text-mid)" }}>Backend Connected</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
