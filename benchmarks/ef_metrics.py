from __future__ import annotations


def hit_rate_at_k(labels_sorted: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = labels_sorted[:k]
    return float(sum(top) / len(top)) if top else 0.0


def topk_active_fraction(labels_sorted: list[int], k: int) -> float:
    return hit_rate_at_k(labels_sorted, k)


def enrichment_factor(labels_sorted: list[int], top_fraction: float) -> float:
    n = len(labels_sorted)
    if n == 0:
        return 0.0
    top_n = max(1, int(round(n * top_fraction)))
    top_hits = sum(labels_sorted[:top_n])
    total_hits = sum(labels_sorted)
    if total_hits == 0:
        return 0.0
    random_expectation = total_hits / n
    top_frac = top_hits / top_n
    return float(top_frac / max(random_expectation, 1e-12))


def ef1(labels_sorted: list[int]) -> float:
    return enrichment_factor(labels_sorted, 0.01)


def ef5(labels_sorted: list[int]) -> float:
    return enrichment_factor(labels_sorted, 0.05)

