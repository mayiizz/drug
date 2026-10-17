from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_validity_vs_step(metrics_path: Path, out_dir: Path) -> Path | None:
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(payload, list) or not payload:
        return None
    steps = [int(r.get("step", i + 1)) for i, r in enumerate(payload)]
    vals = [float(r.get("raw_valid_fraction", 0.0)) for r in payload]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, vals, linewidth=2.0)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Raw validity")
    ax.set_title("Prior validity vs training step")
    ax.grid(alpha=0.3)
    out = _ensure_dir(out_dir) / "validity_vs_training_step.png"
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def plot_docking_hist(candidates: list[dict[str, Any]], out_dir: Path) -> Path:
    vals = [float(c["docking_score"]) for c in candidates if isinstance(c.get("docking_score"), (int, float))]
    fig, ax = plt.subplots(figsize=(7, 4))
    if vals:
        ax.hist(vals, bins=min(20, max(5, len(vals))))
    else:
        ax.text(0.5, 0.5, "No docking scores", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Docking score (kcal/mol)")
    ax.set_ylabel("Count")
    ax.set_title("Docking score distribution")
    out = _ensure_dir(out_dir) / "docking_score_histogram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def plot_kd_hist(candidates: list[dict[str, Any]], out_dir: Path) -> Path:
    vals = [
        float(c.get("kd_nm", c.get("binding_nm")))
        for c in candidates
        if isinstance(c.get("kd_nm", c.get("binding_nm")), (int, float))
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    if vals:
        ax.hist(vals, bins=min(20, max(5, len(vals))))
    else:
        ax.text(0.5, 0.5, "No KD / binding nM values", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Kd (nM)")
    ax.set_ylabel("Count")
    ax.set_title("KD distribution")
    out = _ensure_dir(out_dir) / "kd_distribution_histogram.png"
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def plot_qed_vs_sa(candidates: list[dict[str, Any]], out_dir: Path) -> Path:
    xs: list[float] = []
    ys: list[float] = []
    for c in candidates:
        admet = c.get("admet") or {}
        q = admet.get("drug_likeness")
        s = admet.get("sa_score")
        if isinstance(q, (int, float)) and isinstance(s, (int, float)):
            xs.append(float(q))
            ys.append(float(s))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys, alpha=0.8)
    ax.set_xlabel("QED")
    ax.set_ylabel("SA")
    ax.set_title("QED vs SA")
    out = _ensure_dir(out_dir) / "qed_vs_sa_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def plot_novelty_vs_similarity(target_metrics: dict[str, Any], out_dir: Path) -> Path:
    sims = target_metrics.get("nearest_known_similarity", {})
    vals = [float(sims.get("mean", 0.0)), float(sims.get("median", 0.0)), float(sims.get("top10_mean", 0.0))]
    novelty = [1.0 - v for v in vals]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(vals, novelty, s=80)
    labels = ["mean", "median", "top10"]
    for x, y, label in zip(vals, novelty, labels):
        ax.annotate(label, (x, y))
    ax.set_xlabel("Similarity to known inhibitors")
    ax.set_ylabel("Novelty proxy (1-similarity)")
    ax.set_title("Novelty vs similarity")
    out = _ensure_dir(out_dir) / "novelty_vs_similarity_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def plot_ablation_bars(ablation_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [str(r.get("scenario", "")) for r in ablation_rows]
    vals = [float(r.get("returned_candidates", 0.0)) for r in ablation_rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, vals)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Returned candidates")
    ax.set_title("Ablation comparison")
    out = _ensure_dir(out_dir) / "ablation_comparison_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def plot_survival_funnel(runtime_metrics: dict[str, Any], out_dir: Path) -> Path:
    funnel = runtime_metrics.get("survival_funnel", {}) or {}
    labels = ["generated", "valid", "docked", "ranked", "returned"]
    vals = [int(funnel.get(k, 0)) for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals)
    ax.set_ylabel("Count")
    ax.set_title("Survival funnel")
    out = _ensure_dir(out_dir) / "survival_funnel_plot.png"
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out
