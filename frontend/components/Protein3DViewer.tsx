"use client";

import React, { useEffect, useRef, useState } from "react";
import { Box, Layers, MapPin, RotateCcw, Loader2 } from "lucide-react";
import type { GLViewer } from "3dmol/build/3Dmol.js";

export type ViewMode = "secondary" | "surface" | "active" | "reset";

interface Protein3DViewerProps {
  pdbText: string;
  uniprotId: string;
  /** UniProt active / binding site residue numbers to highlight */
  activeSiteResidues?: number[];
  className?: string;
}

/** AlphaFold pLDDT → confidence color (official palette). */
function plddtColor(atom: { b?: number }): string {
  const b = atom.b ?? 0;
  if (b > 90) return "#0053D6";
  if (b > 70) return "#65CBF3";
  if (b > 50) return "#FFDB13";
  return "#FF7D45";
}

/** Secondary-structure cartoon colors. */
function ssColor(atom: { ss?: string }): string {
  if (atom.ss === "h") return "#FF6B6B";
  if (atom.ss === "s") return "#4ECDC4";
  return "#FFE66D";
}

const CONTROLS: { mode: ViewMode; label: string; icon: typeof Box }[] = [
  { mode: "secondary", label: "Secondary Structure", icon: Layers },
  { mode: "surface", label: "Surface View", icon: Box },
  { mode: "active", label: "Active Sites", icon: MapPin },
  { mode: "reset", label: "Reset View", icon: RotateCcw },
];

export default function Protein3DViewer({
  pdbText,
  uniprotId,
  activeSiteResidues = [],
  className = "",
}: Protein3DViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<GLViewer | null>(null);
  const [mode, setMode] = useState<ViewMode>("secondary");
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Create viewer + load model once per PDB
  useEffect(() => {
    let cancelled = false;
    let viewer: GLViewer | null = null;

    const init = async () => {
      if (!containerRef.current || !pdbText) return;
      setReady(false);
      setError(null);

      try {
        const mod = await import("3dmol/build/3Dmol.js");
        const $3Dmol = mod.default ?? mod;
        if (cancelled || !containerRef.current) return;

        containerRef.current.innerHTML = "";
        viewer = $3Dmol.createViewer(containerRef.current, {
          backgroundColor: "#0b1824",
        });
        viewer.addModel(pdbText, "pdb");
        viewer.setStyle({}, { cartoon: { colorfunc: ssColor } });
        viewer.zoomTo();
        viewer.render();
        viewerRef.current = viewer;
        setMode("secondary");
        setReady(true);
      } catch (e: unknown) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to initialize 3D viewer.");
        }
      }
    };

    init();

    return () => {
      cancelled = true;
      try {
        viewer?.removeAllModels();
        viewer?.clear();
      } catch {
        /* canvas may already be gone */
      }
      viewerRef.current = null;
    };
  }, [pdbText]);

  // Apply style when mode changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !ready) return;

    const apply = async () => {
      try {
        viewer.removeAllSurfaces?.();
        viewer.setStyle({}, {});

        if (mode === "surface") {
          viewer.setStyle({}, { cartoon: { color: "white", opacity: 0.35 } });
          const mod = await import("3dmol/build/3Dmol.js");
          const $3Dmol = mod.default ?? mod;
          await viewer.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.85, color: "spectrum" }, {});
        } else if (mode === "active") {
          // Base cartoon by pLDDT; highlight UniProt active/binding sites when available
          viewer.setStyle({}, { cartoon: { colorfunc: plddtColor } });
          if (activeSiteResidues.length > 0) {
            viewer.addStyle(
              { resi: activeSiteResidues },
              { stick: { radius: 0.25, color: "#FF4D6D" }, sphere: { scale: 0.35, color: "#FF4D6D" } }
            );
          } else {
            viewer.addStyle({ b: { $lt: 50 } }, { stick: { radius: 0.12, color: "#FF7D45" } });
          }
        } else {
          // secondary + reset both use SS coloring; reset also re-zooms
          viewer.setStyle({}, { cartoon: { colorfunc: ssColor } });
          if (mode === "reset") {
            viewer.zoomTo();
            setMode("secondary");
          }
        }

        viewer.render();
      } catch {
        /* ignore surface races during unmount */
      }
    };

    apply();
  }, [mode, ready, activeSiteResidues]);

  // Keep canvas sized to container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      viewerRef.current?.resize?.();
      viewerRef.current?.render?.();
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ready]);

  return (
    <div className={className}>
      <div className="flex flex-wrap gap-2 mb-3">
        {CONTROLS.map(({ mode: m, label, icon: Icon }) => {
          const active = mode === m || (m === "secondary" && mode === "reset");
          return (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              disabled={!ready}
              className="px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all disabled:opacity-50"
              style={{
                background: active ? "rgba(63,184,232,0.15)" : "var(--bg)",
                border: `1px solid ${active ? "rgba(63,184,232,0.4)" : "var(--border)"}`,
                color: active ? "var(--blue-mid)" : "var(--text-mid)",
              }}
            >
              <Icon size={13} />
              {label}
            </button>
          );
        })}
      </div>

      <div
        className="relative rounded-2xl overflow-hidden"
        style={{
          border: "1px solid var(--border)",
          background: "#0b1824",
          minHeight: 420,
        }}
      >
        <div
          ref={containerRef}
          className="w-full"
          style={{ height: 420, position: "relative" }}
          aria-label={`3D structure viewer for ${uniprotId}`}
        />

        {!ready && !error && (
          <div className="absolute inset-0 flex items-center justify-center gap-2 text-sm" style={{ color: "#7dd9f7" }}>
            <Loader2 size={16} className="animate-spin" />
            Rendering structure…
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center p-6 text-sm text-center" style={{ color: "#f87171" }}>
            {error}
          </div>
        )}

        {ready && mode === "active" && (
          <div
            className="absolute bottom-3 left-3 right-3 flex flex-wrap gap-3 text-[10px] font-medium px-3 py-2 rounded-lg"
            style={{ background: "rgba(0,0,0,0.55)", color: "#e2e8f0" }}
          >
            {activeSiteResidues.length > 0 ? (
              <span>
                <span style={{ color: "#FF4D6D" }}>●</span>{" "}
                {activeSiteResidues.length} UniProt active/binding residue
                {activeSiteResidues.length === 1 ? "" : "s"} highlighted · cartoon = pLDDT
              </span>
            ) : (
              <>
                <span><span style={{ color: "#0053D6" }}>●</span> Very high (&gt;90)</span>
                <span><span style={{ color: "#65CBF3" }}>●</span> Confident (70–90)</span>
                <span><span style={{ color: "#FFDB13" }}>●</span> Low (50–70)</span>
                <span><span style={{ color: "#FF7D45" }}>●</span> Very low (&lt;50)</span>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
