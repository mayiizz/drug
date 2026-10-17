from __future__ import annotations

from dataclasses import dataclass

from benchmarks.utils import morgan_fp, murcko_scaffold, tanimoto


@dataclass(slots=True)
class NoveltyStats:
    mean_similarity_to_actives: float
    fraction_below_threshold: float
    scaffold_novelty_ratio: float
    nearest_neighbor_similarity: dict[str, float]


def novelty_analysis(
    generated_smiles: list[str],
    active_smiles: list[str],
    novelty_threshold: float = 0.4,
) -> NoveltyStats:
    active_fps = {s: morgan_fp(s) for s in active_smiles}
    active_scafs = {murcko_scaffold(s) for s in active_smiles}
    nearest: dict[str, float] = {}
    sims: list[float] = []
    nov_scaf = 0
    for s in generated_smiles:
        fp = morgan_fp(s)
        if fp is None:
            continue
        best = max((tanimoto(fp, afp) for afp in active_fps.values()), default=0.0)
        nearest[s] = best
        sims.append(best)
        if murcko_scaffold(s) not in active_scafs:
            nov_scaf += 1
    mean_sim = float(sum(sims) / len(sims)) if sims else 0.0
    below = float(sum(1 for x in sims if x < novelty_threshold) / len(sims)) if sims else 0.0
    scaf_ratio = float(nov_scaf / len(generated_smiles)) if generated_smiles else 0.0
    return NoveltyStats(mean_sim, below, scaf_ratio, nearest)

