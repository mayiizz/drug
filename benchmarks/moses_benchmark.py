from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from benchmarks.metrics import summarize_smiles_properties
from benchmarks.moses_metrics import MosesEvaluator
from benchmarks.utils import morgan_fp, preprocess_smiles, tanimoto


def _internal_diversity(smiles_list: list[str]) -> float:
    fps = [morgan_fp(s) for s in smiles_list]
    vals: list[float] = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if fps[i] is None or fps[j] is None:
                continue
            vals.append(1.0 - tanimoto(fps[i], fps[j]))
    return float(sum(vals) / len(vals)) if vals else 0.0


def _scaffold_diversity(smiles_list: list[str]) -> float:
    scafs: set[str] = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            s = Chem.MolToSmiles(sc) if sc is not None else ""
            if s:
                scafs.add(s)
        except Exception:
            continue
    return float(len(scafs) / max(1, len(smiles_list)))


def evaluate_moses(
    generated_smiles: list[str],
    train_smiles: list[str],
    *,
    out_path: Path = Path("results/benchmarks/moses_metrics.json"),
    max_train_reference: int = 2000,
) -> dict[str, Any]:
    gen = preprocess_smiles(generated_smiles)
    train_raw = train_smiles[: max(0, int(max_train_reference))]
    train = preprocess_smiles(train_raw)
    evaluator = MosesEvaluator()
    core = evaluator.evaluate(gen, train_smiles=train, test_smiles=train[: min(2000, len(train))])
    props = summarize_smiles_properties(gen)
    out: dict[str, Any] = {
        "n_generated_input": len(generated_smiles),
        "n_generated_valid_unique": len(gen),
        "n_train_reference": len(train),
        "max_train_reference_cap": int(max_train_reference),
        "validity": float(core.get("validity", 0.0)),
        "uniqueness": float(core.get("uniqueness", 0.0)),
        "novelty": float(core.get("novelty", 0.0)),
        "internal_diversity": _internal_diversity(gen),
        "scaffold_diversity": _scaffold_diversity(gen),
        "fcd": float(core.get("fcd", 0.0)),
        "average_qed": float(props.get("avg_qed", 0.0)),
        "average_sa": float(props.get("avg_sa", 0.0)),
        "mw_distribution": props.get("mw_stats", {}),
        "logp_distribution": props.get("logp_stats", {}),
        "qed_distribution": props.get("qed_stats", {}),
        "sa_distribution": props.get("sa_stats", {}),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
