from __future__ import annotations

from collections import deque

from rdkit import Chem
from rdkit.Chem import Descriptors

from models.generation.safe_gpt.config import DiversityConfig
from models.generation.safe_gpt.scaffold import murcko_scaffold_smiles
from models.generation.safe_gpt.utils import morgan_fp, tanimoto


class DiversityMemory:
    def __init__(self, max_size: int = 256):
        self.items: deque[object] = deque(maxlen=max_size)
        self.scaffolds: deque[str] = deque(maxlen=max_size)

    def add(self, fp, scaffold: str) -> None:
        self.items.append(fp)
        self.scaffolds.append(scaffold)


def select_diverse_top(
    smiles_reward: dict[str, float],
    seed_fp,
    n: int,
    cfg: DiversityConfig,
    memory: DiversityMemory | None = None,
) -> list[str]:
    sorted_smiles = sorted(smiles_reward.keys(), key=lambda s: smiles_reward[s], reverse=True)
    selected: list[str] = []
    selected_fps: list[object] = []
    selected_scafs: set[str] = set()

    for smi in sorted_smiles:
        if len(selected) >= n:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = morgan_fp(smi)
        if fp is None:
            continue

        if any(a.GetFormalCharge() != 0 or a.GetAtomicNum() in cfg.forbidden_atomic_nums for a in mol.GetAtoms()):
            continue
        if tanimoto(fp, seed_fp) >= cfg.seed_similarity_max:
            continue
        if any(tanimoto(fp, sfp) > cfg.tanimoto_threshold for sfp in selected_fps):
            continue
        if memory is not None and cfg.use_rolling_memory:
            if any(tanimoto(fp, mfp) > cfg.tanimoto_threshold for mfp in memory.items):
                continue

        mw = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
        if not (cfg.mw_min <= mw <= cfg.mw_max):
            continue

        scaf = murcko_scaffold_smiles(smi)
        if scaf and scaf in selected_scafs:
            # mild scaffold-collapse prevention
            continue

        selected.append(smi)
        selected_fps.append(fp)
        if scaf:
            selected_scafs.add(scaf)
        if memory is not None and cfg.use_rolling_memory:
            memory.add(fp, scaf)
    return selected

