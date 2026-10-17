from __future__ import annotations

from benchmarks.utils import morgan_fp, murcko_scaffold, tanimoto


def nearest_active_similarity(generated: list[str], actives: list[str]) -> dict[str, float]:
    active_fps = [morgan_fp(s) for s in actives]
    out: dict[str, float] = {}
    for g in generated:
        gfp = morgan_fp(g)
        if gfp is None:
            out[g] = 0.0
            continue
        out[g] = max((tanimoto(gfp, afp) for afp in active_fps if afp is not None), default=0.0)
    return out


def scaffold_similarity(generated: list[str], actives: list[str]) -> dict[str, float]:
    active_scaf_fps = [morgan_fp(murcko_scaffold(a)) for a in actives]
    out: dict[str, float] = {}
    for g in generated:
        gsc = murcko_scaffold(g)
        gfp = morgan_fp(gsc) if gsc else None
        out[g] = max((tanimoto(gfp, afp) for afp in active_scaf_fps if afp is not None), default=0.0)
    return out


def similarity_histogram(values: list[float], bins: int = 10) -> dict[str, list[float]]:
    if not values:
        return {"bin_edges": [], "counts": []}
    lo, hi = 0.0, 1.0
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, max(0, int((v - lo) / max(step, 1e-9))))
        counts[idx] += 1
    edges = [lo + i * step for i in range(bins + 1)]
    return {"bin_edges": edges, "counts": counts}

