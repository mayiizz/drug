"""Diversity-aware filtering and rolling memory for REINVENT."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from rdkit import Chem
from rdkit.Chem import Descriptors

from models.generation.reinvent4.config import DiversityConfig
from rdkit.Chem.Scaffolds import MurckoScaffold
from models.generation.reinvent4.utils import morgan_fp, tanimoto


@dataclass(slots=True)
class DiversityEntry:
    smiles: str
    scaffold: str
    fp: object
    reward: float | None = None
    docking_success: bool | None = None
    vina_score: float | None = None


@dataclass(slots=True)
class DiversityStats:
    buffer_occupancy: int
    rejected_count: int
    accepted_count: int
    mean_tanimoto: float
    unique_scaffolds: int
    scaffold_entropy: float
    scaffold_frequency: dict[str, int] = field(default_factory=dict)


class RollingDiversityMemory:
    def __init__(self, buffer_size: int = 50, tanimoto_threshold: float = 0.80):
        self.buffer_size = int(buffer_size)
        self.tanimoto_threshold = float(tanimoto_threshold)
        self.entries: deque[DiversityEntry] = deque(maxlen=self.buffer_size)
        self.fp_cache: dict[str, object] = {}
        self.scaffold_counter: Counter[str] = Counter()
        self.rejected_count: int = 0
        self.accepted_count: int = 0

    def _scaffold(self, smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(sc) if sc is not None else ""
        except Exception:
            return ""

    def _fp(self, smi: str):
        if smi in self.fp_cache:
            return self.fp_cache[smi]
        fp = morgan_fp(smi)
        if fp is not None:
            self.fp_cache[smi] = fp
        return fp

    def _mean_tanimoto(self, fp) -> float:
        sims = [tanimoto(fp, e.fp) for e in self.entries if e.fp is not None]
        return float(sum(sims) / len(sims)) if sims else 0.0

    def should_accept(self, smi: str, *, threshold: float | None = None) -> tuple[bool, str]:
        fp = self._fp(smi)
        if fp is None:
            return False, "invalid_fingerprint"
        thr = float(self.tanimoto_threshold if threshold is None else threshold)

        # molecule-level repetition
        for e in self.entries:
            if tanimoto(fp, e.fp) > thr:
                return False, "high_tanimoto_molecule"

        # scaffold-level repetition
        sc = self._scaffold(smi)
        if sc:
            sc_fp = self._fp(sc)
            for e in self.entries:
                if e.scaffold and sc_fp is not None:
                    esc_fp = self._fp(e.scaffold)
                    if esc_fp is not None and tanimoto(sc_fp, esc_fp) > thr:
                        return False, "high_tanimoto_scaffold"
        return True, "accepted"

    def add(
        self,
        smi: str,
        *,
        reward: float | None = None,
        docking_success: bool | None = None,
        vina_score: float | None = None,
    ) -> None:
        fp = self._fp(smi)
        if fp is None:
            return
        sc = self._scaffold(smi)
        # deque maxlen auto-evicts oldest; update scaffold counter manually.
        if len(self.entries) == self.entries.maxlen and len(self.entries) > 0:
            old = self.entries[0]
            if old.scaffold:
                self.scaffold_counter[old.scaffold] -= 1
                if self.scaffold_counter[old.scaffold] <= 0:
                    del self.scaffold_counter[old.scaffold]
        self.entries.append(
            DiversityEntry(
                smiles=smi,
                scaffold=sc,
                fp=fp,
                reward=reward,
                docking_success=docking_success,
                vina_score=vina_score,
            )
        )
        if sc:
            self.scaffold_counter[sc] += 1
        self.accepted_count += 1

    def reject(self) -> None:
        self.rejected_count += 1

    def scaffold_entropy(self) -> float:
        total = sum(self.scaffold_counter.values())
        if total <= 0:
            return 0.0
        entropy = 0.0
        for c in self.scaffold_counter.values():
            p = c / total
            entropy -= p * math.log(max(p, 1e-12))
        return float(entropy)

    def stats(self) -> DiversityStats:
        mean_sim = 0.0
        if len(self.entries) > 1:
            vals = []
            fps = [e.fp for e in self.entries if e.fp is not None]
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    vals.append(tanimoto(fps[i], fps[j]))
            mean_sim = float(sum(vals) / len(vals)) if vals else 0.0
        return DiversityStats(
            buffer_occupancy=len(self.entries),
            rejected_count=self.rejected_count,
            accepted_count=self.accepted_count,
            mean_tanimoto=mean_sim,
            unique_scaffolds=len(self.scaffold_counter),
            scaffold_entropy=self.scaffold_entropy(),
            scaffold_frequency=dict(self.scaffold_counter),
        )

    def state_dict(self) -> dict:
        return {
            "buffer_size": self.buffer_size,
            "tanimoto_threshold": self.tanimoto_threshold,
            "entries": [
                {
                    "smiles": e.smiles,
                    "scaffold": e.scaffold,
                    "reward": e.reward,
                    "docking_success": e.docking_success,
                    "vina_score": e.vina_score,
                }
                for e in self.entries
            ],
            "rejected_count": self.rejected_count,
            "accepted_count": self.accepted_count,
        }

    def load_state_dict(self, d: dict) -> None:
        self.buffer_size = int(d.get("buffer_size", self.buffer_size))
        self.tanimoto_threshold = float(d.get("tanimoto_threshold", self.tanimoto_threshold))
        self.entries = deque(maxlen=self.buffer_size)
        self.scaffold_counter = Counter()
        for x in d.get("entries", []):
            smi = x.get("smiles", "")
            fp = self._fp(smi)
            if not smi or fp is None:
                continue
            sc = x.get("scaffold") or self._scaffold(smi)
            self.entries.append(
                DiversityEntry(
                    smiles=smi,
                    scaffold=sc,
                    fp=fp,
                    reward=x.get("reward"),
                    docking_success=x.get("docking_success"),
                    vina_score=x.get("vina_score"),
                )
            )
            if sc:
                self.scaffold_counter[sc] += 1
        self.rejected_count = int(d.get("rejected_count", 0))
        self.accepted_count = int(d.get("accepted_count", len(self.entries)))


def select_diverse_top(
    smiles_reward: dict[str, float],
    seed_fp,
    n: int,
    div: DiversityConfig | None = None,
) -> list[str]:
    """
    Greedy diversity selection: sort by reward, add molecules with low mutual Tanimoto.
    """
    if div is None:
        div = DiversityConfig()

    sorted_smiles = sorted(
        smiles_reward.keys(),
        key=lambda s: smiles_reward[s],
        reverse=True,
    )
    selected: list[str] = []
    selected_fps: list = []

    for smi in sorted_smiles:
        if len(selected) >= n:
            break
        fp = morgan_fp(smi)
        if fp is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        skip_atom = False
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0 or atom.GetAtomicNum() in div.forbidden_atomic_nums:
                skip_atom = True
                break
        if skip_atom:
            continue

        sim_to_seed = tanimoto(fp, seed_fp)
        if sim_to_seed >= div.seed_tanimoto_max:
            continue

        too_similar = any(tanimoto(fp, sfp) > div.intra_tanimoto_max for sfp in selected_fps)
        if too_similar:
            continue

        mw = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
        if not (div.mw_min <= mw <= div.mw_max):
            continue

        if div.use_scaffold_diversity:
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                sc_smi = Chem.MolToSmiles(scaf) if scaf else ""
            except Exception:
                sc_smi = ""
            scafs = set()
            for ex in selected:
                em = Chem.MolFromSmiles(ex)
                if em is None:
                    continue
                try:
                    sc = MurckoScaffold.GetScaffoldForMol(em)
                    if sc:
                        scafs.add(Chem.MolToSmiles(sc))
                except Exception:
                    pass
            if sc_smi and sc_smi in scafs and len(selected) > 0:
                continue

        selected.append(smi)
        selected_fps.append(fp)

    return selected
