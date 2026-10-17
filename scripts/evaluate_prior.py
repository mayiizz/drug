from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.generation.reinvent4.config import Reinvent4Config
from models.generation.reinvent4.checkpoint_utils import (
    checkpoint_metadata,
    resolve_best_prior_checkpoint,
    state_param_count,
    torch_load_checkpoint,
    unwrap_model_state_dict,
)
from models.generation.reinvent4.prior import build_and_load_prior
from models.generation.reinvent4.utils import safe_sa
from models.generation.reinvent4.vocab import SMILESVocabulary

logger = logging.getLogger("evaluate_prior")


def _load_train_set(path: Path) -> set[str]:
    import pandas as pd

    if not path.exists():
        return set()
    df = pd.read_csv(path)
    col = next((c for c in ("smiles", "canonical_smiles", "SMILES") if c in df.columns), None)
    if col is None:
        return set()
    out: set[str] = set()
    for x in df[col].dropna().tolist():
        m = Chem.MolFromSmiles(str(x))
        if m is None:
            continue
        out.add(Chem.MolToSmiles(m, canonical=True))
    return out


def evaluate_prior(
    *,
    samples: int,
    temperature: float,
    top_k: int,
    out_path: Path,
    train_csv: Path | None,
) -> dict:
    cfg = Reinvent4Config()
    vocab = SMILESVocabulary()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_and_load_prior(vocab, device, cfg)
    resolved = resolve_best_prior_checkpoint()
    state = torch_load_checkpoint(resolved.path, device)
    sd = unwrap_model_state_dict(state)
    model.load_state_dict(sd, strict=False)
    logger.info("[PRIOR] checkpoint=%s", resolved.path)
    logger.info("[PRIOR] params=%d", state_param_count(sd))
    logger.info("[PRIOR] vocab=%d", len(vocab))
    meta = checkpoint_metadata(state)
    if meta:
        logger.info("[PRIOR] metadata=%s", meta)
    if resolved.is_bootstrap_fallback:
        logger.warning("[PRIOR] warning: using bootstrap prior weights")
    train_set = _load_train_set(train_csv) if train_csv is not None else set()

    smiles, _ = model.sample(
        vocab,
        batch_size=samples,
        max_len=72,
        temperature=temperature,
        top_k=top_k,
        device=device,
        greedy=False,
    )
    stats = dict(getattr(model, "last_sample_stats", {}) or {})
    sanitized_valid_fraction = float(len(smiles) / max(1, int(stats.get("decoded", len(smiles)))))
    unique_fraction = float(len(set(smiles)) / max(1, len(smiles)))
    novelty_fraction = (
        float(sum(1 for s in smiles if s not in train_set) / max(1, len(smiles)))
        if train_set
        else 0.0
    )

    mw_vals: list[float] = []
    logp_vals: list[float] = []
    qed_vals: list[float] = []
    sa_vals: list[float] = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        mw_vals.append(float(Descriptors.MolWt(m)))
        logp_vals.append(float(Descriptors.MolLogP(m)))
        qed_vals.append(float(QED.qed(m)))
        sa_vals.append(float(safe_sa(m)))

    report = {
        "prior_file": str(resolved.path),
        "sample_settings": {
            "samples": samples,
            "temperature": temperature,
            "top_k": top_k,
        },
        "raw_valid_fraction": float(stats.get("raw_valid_fraction", 0.0)),
        "sanitized_valid_fraction": sanitized_valid_fraction,
        "decoded": int(stats.get("decoded", len(smiles))),
        "raw_valid": int(stats.get("raw_valid", len(smiles))),
        "raw_invalid": int(stats.get("raw_invalid", 0)),
        "sanitize_failures": int(stats.get("sanitize_failures", 0)),
        "returned_valid": len(smiles),
        "uniqueness": unique_fraction,
        "novelty": novelty_fraction,
        "avg_mw": float(sum(mw_vals) / max(1, len(mw_vals))),
        "avg_logp": float(sum(logp_vals) / max(1, len(logp_vals))),
        "avg_qed": float(sum(qed_vals) / max(1, len(qed_vals))),
        "avg_sa": float(sum(sa_vals) / max(1, len(sa_vals))),
        "sample_preview": smiles[:20],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CURA prior quality")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--train-csv", type=Path, default=Path("data/chembl_100k.csv"))
    p.add_argument("--out", type=Path, default=Path("results/prior_eval.json"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = _parse_args()
    report = evaluate_prior(
        samples=int(args.samples),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        out_path=args.out,
        train_csv=args.train_csv,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

