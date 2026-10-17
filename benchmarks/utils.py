from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def stable_json_dumps(obj: Any) -> str:
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(stable_json_dumps(obj).encode("utf-8")).hexdigest()


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def strip_salts(smiles: str) -> str:
    parts = [p for p in smiles.split(".") if p]
    if not parts:
        return ""
    parts = sorted(parts, key=len, reverse=True)
    return canonicalize_smiles(parts[0])


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc is not None else ""
    except Exception:
        return ""


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return float(TanimotoSimilarity(fp1, fp2))


def preprocess_smiles(smiles_list: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in smiles_list:
        c = canonicalize_smiles(strip_salts(s))
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def preprocess_with_scaffold_dedup(smiles_list: list[str]) -> list[str]:
    out: list[str] = []
    seen_scaf: set[str] = set()
    for s in preprocess_smiles(smiles_list):
        sc = murcko_scaffold(s)
        key = sc or s
        if key in seen_scaf:
            continue
        seen_scaf.add(key)
        out.append(s)
    return out


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

