from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

from benchmarks.ablation_benchmark import AblationRunner
from benchmarks.metrics import (
    aggregate_distribution_stats,
    load_corpus_smiles,
    locate_chembl_corpus,
    summarize_smiles_properties,
)
from benchmarks.moses_benchmark import evaluate_moses
from benchmarks.plotting import (
    plot_ablation_bars,
    plot_docking_hist,
    plot_kd_hist,
    plot_novelty_vs_similarity,
    plot_qed_vs_sa,
    plot_survival_funnel,
    plot_validity_vs_step,
)
from benchmarks.runtime_benchmark import evaluate_runtime
from benchmarks.target_benchmark import evaluate_target
from models.generation.reinvent4.config import Reinvent4Config
from models.runtime.cura_pipeline import CuraPipeline

logger = logging.getLogger("run_benchmarks")


def _resolve_protein_sequence(target_name: str) -> str:
    try:
        from utils.protein_fetcher import fetch_protein_sequence

        data = fetch_protein_sequence(target_name) or fetch_protein_sequence("EGFR")
        if data and isinstance(data, dict) and data.get("sequence"):
            return str(data["sequence"])
    except Exception as e:
        logger.warning("[bench] protein fetch failed for %s: %s", target_name, e)
    return "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLE"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CURA scientific benchmarks.")
    parser.add_argument("--target", default="EGFR")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--mode", action="append", choices=["prior_only", "rl"], default=["prior_only"])
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--moses", action="store_true")
    return parser.parse_args(argv)


def _build_report(
    *,
    out_dir: Path,
    mode: str,
    candidates: list[dict[str, Any]],
    moses: dict[str, Any] | None,
    target_metrics: dict[str, Any],
    runtime_metrics: dict[str, Any] | None,
    ablation_rows: list[dict[str, Any]] | None,
) -> Path:
    def _dock_sort_key(row: dict[str, Any]) -> float:
        v = row.get("docking_score")
        return float(v) if isinstance(v, (int, float)) else 99.0

    top = sorted(candidates, key=_dock_sort_key)[:10]
    lines = [
        "# CURA Benchmark Report",
        "",
        f"- Mode: `{mode}`",
        f"- Candidates: `{len(candidates)}`",
        "",
        "## Summary Metrics",
        "",
    ]
    if moses:
        lines += [
            "| Metric | Value |",
            "|---|---:|",
            f"| Validity | {moses.get('validity', 0.0):.4f} |",
            f"| Uniqueness | {moses.get('uniqueness', 0.0):.4f} |",
            f"| Novelty | {moses.get('novelty', 0.0):.4f} |",
            f"| Internal diversity | {moses.get('internal_diversity', 0.0):.4f} |",
            "",
        ]
    dock = target_metrics.get("docking_score_distribution", {})
    kd = target_metrics.get("kd_distribution_nm", {})
    lines += [
        "| Distribution | Mean | Median | Std | Top 10% Mean |",
        "|---|---:|---:|---:|---:|",
        f"| Docking score | {dock.get('mean', 0.0):.4f} | {dock.get('median', 0.0):.4f} | {dock.get('std', 0.0):.4f} | {dock.get('top10_mean', 0.0):.4f} |",
        f"| KD (nM) | {kd.get('mean', 0.0):.4f} | {kd.get('median', 0.0):.4f} | {kd.get('std', 0.0):.4f} | {kd.get('top10_mean', 0.0):.4f} |",
        "",
        "## Top Molecules",
        "",
        "| SMILES | Docking | KD (nM) | Score |",
        "|---|---:|---:|---:|",
    ]
    def _fmt3(v: object) -> str:
        return f"{float(v):.3f}" if isinstance(v, (int, float)) else "—"

    for row in top:
        ds = row.get("docking_score")
        kd = row.get("kd_nm", row.get("binding_nm"))
        sc = row.get("score")
        if not isinstance(sc, (int, float)):
            sc = row.get("drug_score")
        if not isinstance(sc, (int, float)):
            val = row.get("validation") if isinstance(row.get("validation"), dict) else {}
            sc = val.get("composite_score") if isinstance(val, dict) else None
        smi = str(row.get("smiles", "")).strip()
        lines.append(f"| `{smi}` | {_fmt3(ds)} | {_fmt3(kd)} | {_fmt3(sc)} |")
    lines += ["", "## Plots", ""]
    for png in sorted((out_dir / "plots").glob("*.png")):
        lines.append(f"- ![{png.name}](plots/{png.name})")
    if runtime_metrics:
        lines += ["", "## Runtime", "", f"```json\n{json.dumps(runtime_metrics, indent=2)}\n```"]
    if ablation_rows:
        lines += ["", "## Ablation Rows", "", f"```json\n{json.dumps(ablation_rows, indent=2)}\n```"]
    report = out_dir / "benchmark_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _run_mode(
    mode: str,
    target: str,
    samples: int,
    out_dir: Path,
    run_moses: bool,
    run_runtime: bool,
    run_ablation: bool,
) -> dict[str, Any]:
    protein_sequence = _resolve_protein_sequence(target)
    rcfg = Reinvent4Config()
    rcfg.enable_runtime_rl = mode == "rl"
    pipeline = CuraPipeline(profile="benchmark", reinvent_config=rcfg)
    t0 = time.perf_counter()
    candidates = pipeline.run(
        protein_sequence=protein_sequence,
        target=target,
        num_molecules=samples,
        sample_size=max(samples * 5, 200),
        seed_smiles=None,
    )
    elapsed = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles = [str(c.get("smiles", "")).strip() for c in candidates if c.get("smiles")]
    corpus_path = locate_chembl_corpus(PROJECT_ROOT)
    train_smiles = load_corpus_smiles(corpus_path) if corpus_path else []

    moses = (
        evaluate_moses(smiles, train_smiles, out_path=out_dir / "moses_metrics.json")
        if run_moses
        else None
    )
    target_metrics = evaluate_target(candidates, target=target, out_path=out_dir / "target_metrics.json")
    runtime_metrics = (
        evaluate_runtime(
            sample_size=max(samples * 5, 200),
            returned_candidates=candidates,
            diagnostics=pipeline.last_diagnostics or {},
            elapsed_sec=elapsed,
            out_path=out_dir / "runtime_metrics.json",
        )
        if run_runtime
        else None
    )

    plots_dir = out_dir / "plots"
    plot_docking_hist(candidates, plots_dir)
    plot_kd_hist(candidates, plots_dir)
    plot_qed_vs_sa(candidates, plots_dir)
    plot_novelty_vs_similarity(target_metrics, plots_dir)
    if runtime_metrics:
        plot_survival_funnel(runtime_metrics, plots_dir)
    plot_validity_vs_step(PROJECT_ROOT / "results" / "prior_training_metrics.json", plots_dir)

    ablation_rows = None
    if run_ablation:
        ablation_rows = AblationRunner(
            protein_sequence=protein_sequence, target=target, samples=samples
        ).run(out_csv=out_dir / "ablation_results.csv")
        plot_ablation_bars(ablation_rows, plots_dir)

    summary = {
        "mode": mode,
        "target": target,
        "samples": samples,
        "returned": len(candidates),
        "docking_distribution": target_metrics.get("docking_score_distribution", {}),
        "kd_distribution": target_metrics.get("kd_distribution_nm", {}),
        "props": summarize_smiles_properties(smiles),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    report = _build_report(
        out_dir=out_dir,
        mode=mode,
        candidates=candidates,
        moses=moses,
        target_metrics=target_metrics,
        runtime_metrics=runtime_metrics,
        ablation_rows=ablation_rows,
    )
    return {
        "mode": mode,
        "report": str(report),
        "returned": len(candidates),
        "docking_mean": aggregate_distribution_stats(
            [float(c["docking_score"]) for c in candidates if isinstance(c.get("docking_score"), (int, float))]
        ).get("mean", 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    bench_root = PROJECT_ROOT / "results" / "benchmarks"
    bench_root.mkdir(parents=True, exist_ok=True)

    results = []
    modes = list(dict.fromkeys(args.mode))
    for mode in modes:
        logger.info("[bench] running mode=%s target=%s samples=%d", mode, args.target, args.samples)
        results.append(
            _run_mode(
                mode,
                target=args.target,
                samples=args.samples,
                out_dir=bench_root / mode,
                run_moses=bool(args.moses),
                run_runtime=bool(args.runtime),
                run_ablation=bool(args.ablation),
            )
        )

    (bench_root / "benchmark_runs.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    if modes:
        last_report = bench_root / modes[-1] / "benchmark_report.md"
        if last_report.exists():
            shutil.copy(last_report, bench_root / "benchmark_report.md")
    logger.info("[bench] completed; outputs at %s", bench_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
