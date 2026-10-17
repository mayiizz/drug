from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.MolStandardize import rdMolStandardize

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.generation.reinvent4.agent import SMILESLSTM
from models.generation.reinvent4.checkpoint_utils import utc_now_iso
from models.generation.reinvent4.config import PRIOR_FILENAME, Reinvent4Config
from models.generation.reinvent4.vocab import SMILESVocabulary
from models.generation.reinvent4.utils import safe_sa

logger = logging.getLogger("train_prior")
_SMILES_COLUMNS = ("smiles", "canonical_smiles", "SMILES", "molecule_smiles", "standard_smiles")
_ALLOWED_ATOMIC_NUMS = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def _read_smiles_from_csv(path: Path) -> tuple[list[str], str]:
    import pandas as pd

    df = pd.read_csv(path)
    col = next((c for c in _SMILES_COLUMNS if c in df.columns), None)
    if col is None:
        raise ValueError(f"No SMILES-like column found in {path}")
    return [str(x) for x in df[col].dropna().tolist()], col


def _standardize_smiles(
    s: str,
    chooser: rdMolStandardize.LargestFragmentChooser,
    vocab: SMILESVocabulary,
) -> str | None:
    try:
        raw = (s or "").strip()
        if "*" in raw:
            return None
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            return None
        mol = chooser.choose(mol)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        mw = float(Descriptors.MolWt(mol))
        if not (150.0 <= mw <= 650.0):
            return None
        # Bias corpus toward medicinal heterocycle-rich chemistry used in kinases.
        hetero_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        aromatic_rings = int(
            sum(
                1
                for ring in Chem.GetSymmSSSR(mol)
                if all(mol.GetAtomWithIdx(int(i)).GetIsAromatic() for i in ring)
            )
        )
        if hetero_atoms < 1 or aromatic_rings < 1:
            return None
        can = Chem.MolToSmiles(mol, canonical=True)
        if not can:
            return None
        for atom in mol.GetAtoms():
            if int(atom.GetAtomicNum()) not in _ALLOWED_ATOMIC_NUMS:
                return None
            if atom.GetNumRadicalElectrons() > 0:
                return None
        toks = vocab.tokenize(can)
        if any(tok not in vocab.tok2idx for tok in toks):
            return None
        return can
    except Exception:
        return None


def build_corpus(
    csv_path: Path,
    limit: int | None = None,
) -> tuple[list[str], str, int]:
    vocab = SMILESVocabulary()
    raw, smiles_col = _read_smiles_from_csv(csv_path)
    chooser = rdMolStandardize.LargestFragmentChooser()
    seen: set[str] = set()
    out: list[str] = []
    for s in raw:
        can = _standardize_smiles(s, chooser, vocab)
        if can is None or can in seen:
            continue
        seen.add(can)
        out.append(can)
        if limit is not None and len(out) >= limit:
            break
    return out, smiles_col, len(raw)


@torch.no_grad()
def evaluate_prior_snapshot(
    model: SMILESLSTM,
    vocab: SMILESVocabulary,
    *,
    batch_size: int = 64,
    temperature: float = 0.6,
    top_k: int = 20,
    train_set: set[str] | None = None,
) -> dict:
    device = next(model.parameters()).device
    sampled, _ = model.sample(
        vocab,
        batch_size=batch_size,
        max_len=72,
        temperature=temperature,
        top_k=top_k,
        device=device,
        greedy=False,
    )
    stats = dict(getattr(model, "last_sample_stats", {}) or {})
    decoded = int(stats.get("decoded", len(sampled)))
    sanitized_valid_fraction = float(len(sampled) / max(1, decoded))
    unique_fraction = float(len(set(sampled)) / max(1, len(sampled)))
    if train_set:
        novelty_fraction = float(sum(1 for s in sampled if s not in train_set) / max(1, len(sampled)))
    else:
        novelty_fraction = 0.0

    mw_vals: list[float] = []
    logp_vals: list[float] = []
    qed_vals: list[float] = []
    sa_vals: list[float] = []
    for s in sampled:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        mw_vals.append(float(Descriptors.MolWt(m)))
        logp_vals.append(float(Descriptors.MolLogP(m)))
        qed_vals.append(float(QED.qed(m)))
        sa_vals.append(float(safe_sa(m)))
    stats.update(
        {
            "sampled_returned": len(sampled),
            "sanitized_valid_fraction": sanitized_valid_fraction,
            "unique_fraction": unique_fraction,
            "novelty_fraction": novelty_fraction,
            "avg_mw": float(sum(mw_vals) / max(1, len(mw_vals))),
            "avg_logp": float(sum(logp_vals) / max(1, len(logp_vals))),
            "avg_qed": float(sum(qed_vals) / max(1, len(qed_vals))),
            "avg_sa": float(sum(sa_vals) / max(1, len(sa_vals))),
            "sample_preview": sampled[:20],
        }
    )
    return stats


def _phase_defaults(phase: str) -> tuple[int, int, int]:
    p = (phase or "full").strip().lower()
    if p == "smoke":
        return 5000, 64, 250
    if p == "full":
        return 50000, 128, 500
    raise ValueError(f"Unsupported phase: {phase}")


def _recommendation_for_validity(raw_valid: float) -> str:
    if raw_valid < 0.10:
        return "validity<0.10: continue prior training; grammar still weak."
    if raw_valid < 0.50:
        return "validity<0.50: keep training; do not enable RL."
    if raw_valid < 0.80:
        return "validity>0.50: prior usable; prepare RL warm-start after stabilization."
    return "validity>0.80: prior healthy; RL optimization can be enabled."


def train_prior(
    *,
    smiles_csv: Path,
    checkpoint_dir: Path,
    metrics_path: Path,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    resume: bool,
    corpus_limit: int | None,
    plot_out: Path | None,
    phase: str,
) -> None:
    ckpt_dir = checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "prior_latest.pt"
    best_path = ckpt_dir / "prior_best_validity.pt"

    corpus, smiles_col, raw_count = build_corpus(smiles_csv, limit=corpus_limit)
    logger.info("[DATASET] path=%s", smiles_csv)
    logger.info("[DATASET] total_rows=%d", raw_count)
    logger.info("[DATASET] smiles_column=%s", smiles_col)
    if len(corpus) < 1000:
        logger.warning("Corpus is small (%d). Prior quality may remain weak.", len(corpus))
    train_set = set(corpus)

    cfg = Reinvent4Config()
    vocab = SMILESVocabulary()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SMILESLSTM(
        len(vocab),
        embed_dim=cfg.lstm_embed_dim,
        hidden_dim=cfg.lstm_hidden_dim,
        num_layers=cfg.lstm_num_layers,
        dropout=cfg.lstm_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_step = 0
    best_validity = 0.0
    best_raw_validity_step = 0
    stagnant_evals = 0
    milestones_hit: set[float] = set()
    history: list[dict] = []
    if resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_step = int(state.get("step", 0))
        best_validity = float(state.get("best_raw_validity", 0.0))
        best_raw_validity_step = int(state.get("best_raw_validity_step", 0))
        stale_hist = state.get("milestones_hit", [])
        milestones_hit = set(float(x) for x in stale_hist)
        history = list(state.get("history", []))
        logger.info("Resumed prior training from %s at step=%d", latest_path, start_step)

    logger.info(
        "Prior training started | corpus=%d steps=%d batch=%d lr=%.4g device=%s",
        len(corpus),
        steps,
        batch_size,
        lr,
        device,
    )
    model.train()
    for step in range(start_step, steps):
        batch = random.choices(corpus, k=batch_size)
        losses = []
        for smi in batch:
            ids = vocab.encode(smi)
            if len(ids) < 4:
                continue
            t = torch.tensor([ids], dtype=torch.long, device=device)
            losses.append(-model.log_prob(t).mean())
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % eval_every == 0 or step == 0:
            model.eval()
            stats = evaluate_prior_snapshot(
                model,
                vocab,
                batch_size=256,
                temperature=0.6,
                top_k=20,
                train_set=train_set,
            )
            model.train()
            raw_valid = float(stats.get("raw_valid_fraction", 0.0))
            entry = {
                "step": step + 1,
                "loss": float(loss.item()),
                "phase": phase,
                **stats,
            }
            history.append(entry)
            logger.info("[PRIOR] step=%d", step + 1)
            logger.info("[PRIOR] loss=%.6f", float(loss.item()))
            logger.info("[PRIOR] raw_valid_fraction=%.4f", raw_valid)
            logger.info("[PRIOR] uniqueness=%.4f", float(stats.get("unique_fraction", 0.0)))
            logger.info("[PRIOR] novelty=%.4f", float(stats.get("novelty_fraction", 0.0)))
            logger.info("[PRIOR] sanitize_failures=%d", int(stats.get("sanitize_failures", 0)))

            sample_path = metrics_path.parent / f"prior_samples_step_{step+1:05d}.txt"
            try:
                sample_path.write_text(
                    "\n".join(stats.get("sample_preview", []))
                    if isinstance(stats.get("sample_preview"), list)
                    else "",
                    encoding="utf-8",
                )
            except Exception:
                pass

            warning_messages: list[str] = []
            recent = history[-6:]
            if len(recent) >= 4:
                uniq_recent = [float(h.get("unique_fraction", 0.0)) for h in recent]
                if uniq_recent[-1] < 0.15:
                    warning_messages.append("uniqueness collapse detected")
                val_recent = [float(h.get("raw_valid_fraction", 0.0)) for h in recent]
                if max(val_recent) - min(val_recent) < 0.01:
                    warning_messages.append("validity stagnant for recent evaluations")
                loss_recent = [float(h.get("loss", 0.0)) for h in recent]
                if any((not math.isfinite(v) or v > 5.0 * max(1e-6, loss_recent[0])) for v in loss_recent[1:]):
                    warning_messages.append("training instability detected")
            if float(stats.get("unique_fraction", 0.0)) < 0.10 and float(stats.get("novelty_fraction", 0.0)) < 0.10:
                warning_messages.append("mode collapse detected")
            for w in warning_messages:
                logger.warning("[PRIOR] warning: %s", w)

            for m in (0.10, 0.25, 0.50, 0.80):
                if raw_valid >= m and m not in milestones_hit:
                    milestones_hit.add(m)
                    logger.info("[PRIOR] milestone reached: raw_valid_fraction >= %.2f", m)

            if raw_valid > best_validity:
                stagnant_evals = 0
            else:
                stagnant_evals += 1
                if stagnant_evals >= 6:
                    logger.warning("[PRIOR] warning: validity stagnant for %d evaluations", stagnant_evals)

            rec = _recommendation_for_validity(raw_valid)
            logger.info("[PRIOR] recommendation: %s", rec)
            entry["recommendation"] = rec
            entry["warnings"] = warning_messages

            step_path = ckpt_dir / f"prior_step_{step+1:05d}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_raw_validity": best_validity,
                    "best_raw_validity_step": best_raw_validity_step,
                    "raw_valid_fraction": raw_valid,
                    "uniqueness": float(stats.get("unique_fraction", 0.0)),
                    "novelty": float(stats.get("novelty_fraction", 0.0)),
                    "timestamp_utc": utc_now_iso(),
                    "history": history,
                    "milestones_hit": sorted(milestones_hit),
                    "vocab_size": len(vocab),
                    "prior_filename": PRIOR_FILENAME,
                },
                step_path,
            )
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_raw_validity": max(best_validity, raw_valid),
                    "best_raw_validity_step": (
                        (step + 1) if raw_valid > best_validity else best_raw_validity_step
                    ),
                    "raw_valid_fraction": raw_valid,
                    "uniqueness": float(stats.get("unique_fraction", 0.0)),
                    "novelty": float(stats.get("novelty_fraction", 0.0)),
                    "timestamp_utc": utc_now_iso(),
                    "history": history,
                    "milestones_hit": sorted(milestones_hit),
                    "vocab_size": len(vocab),
                    "prior_filename": PRIOR_FILENAME,
                },
                latest_path,
            )
            if raw_valid >= best_validity:
                best_validity = raw_valid
                best_raw_validity_step = step + 1
                torch.save(
                    {
                        "step": step + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_raw_validity": best_validity,
                        "best_raw_validity_step": best_raw_validity_step,
                        "raw_valid_fraction": raw_valid,
                        "uniqueness": float(stats.get("unique_fraction", 0.0)),
                        "novelty": float(stats.get("novelty_fraction", 0.0)),
                        "timestamp_utc": utc_now_iso(),
                        "history": history,
                        "milestones_hit": sorted(milestones_hit),
                        "vocab_size": len(vocab),
                        "prior_filename": PRIOR_FILENAME,
                    },
                    best_path,
                )
                logger.info("[PRIOR] new best validity checkpoint saved: %s", best_path)
            metrics_path.write_text(
                json.dumps(
                    {
                        "config": {
                            "phase": phase,
                            "steps": steps,
                            "batch_size": batch_size,
                            "lr": lr,
                            "eval_every": eval_every,
                            "csv": str(smiles_csv),
                            "corpus_size": len(corpus),
                        },
                        "best_raw_validity": best_validity,
                        "history": history,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    logger.info("Prior training complete. best_raw_validity=%.3f", best_validity)
    if plot_out is not None:
        try:
            from scripts.plot_prior_training import plot_prior_curves

            plot_prior_curves(metrics_path=metrics_path, out_path=plot_out)
            logger.info("Saved prior training curves to %s", plot_out)
        except Exception as e:
            logger.warning("Could not generate prior training curves: %s", e)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SMILES LSTM prior for CURA")
    p.add_argument(
        "--smiles-csv",
        type=Path,
        default=Path("data/chembl_100k.csv"),
        help="CSV containing a SMILES column",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--metrics-path", type=Path, default=Path("results/prior_training_metrics.json"))
    p.add_argument("--phase", choices=["smoke", "full"], default="full")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--corpus-limit", type=int, default=None)
    p.add_argument("--plot-out", type=Path, default=Path("results/prior_training_curves.png"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = _parse_args()
    def_steps, def_batch, def_eval = _phase_defaults(args.phase)
    train_prior(
        smiles_csv=args.smiles_csv,
        checkpoint_dir=args.checkpoint_dir,
        metrics_path=args.metrics_path,
        steps=int(args.steps if args.steps is not None else def_steps),
        batch_size=int(args.batch_size if args.batch_size is not None else def_batch),
        lr=float(args.lr),
        eval_every=int(args.eval_every if args.eval_every is not None else def_eval),
        resume=bool(args.resume),
        corpus_limit=args.corpus_limit,
        plot_out=args.plot_out,
        phase=str(args.phase),
    )


if __name__ == "__main__":
    main()

