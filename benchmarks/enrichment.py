from __future__ import annotations

from benchmarks.similarity import nearest_active_similarity


def label_actives_by_threshold(
    smiles: list[str],
    activity_map_nm: dict[str, float],
    threshold_nm: float,
) -> list[int]:
    out = []
    for s in smiles:
        v = activity_map_nm.get(s)
        out.append(1 if (v is not None and v <= threshold_nm) else 0)
    return out


def label_actives_by_similarity(
    smiles: list[str],
    known_actives: list[str],
    tanimoto_threshold: float = 0.7,
) -> list[int]:
    sim = nearest_active_similarity(smiles, known_actives)
    return [1 if sim.get(s, 0.0) >= tanimoto_threshold else 0 for s in smiles]

