from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class SummaryStats:
    mean: float
    std: float
    ci95_low: float
    ci95_high: float


def summarize(values: list[float]) -> SummaryStats:
    if not values:
        return SummaryStats(0.0, 0.0, 0.0, 0.0)
    n = len(values)
    mu = sum(values) / n
    var = sum((x - mu) ** 2 for x in values) / (n - 1 if n > 1 else 1)
    sd = math.sqrt(max(var, 0.0))
    sem = sd / math.sqrt(max(n, 1))
    delta = 1.96 * sem
    return SummaryStats(mu, sd, mu - delta, mu + delta)


def welch_ttest(a: list[float], b: list[float]) -> dict[str, float]:
    # Use scipy when available; fallback returns t-stat only.
    if not a or not b:
        return {"t_stat": 0.0, "p_value": 1.0}
    try:
        from scipy.stats import ttest_ind  # type: ignore

        r = ttest_ind(a, b, equal_var=False)
        return {"t_stat": float(r.statistic), "p_value": float(r.pvalue)}
    except Exception:
        sa = summarize(a)
        sb = summarize(b)
        pooled = math.sqrt((sa.std**2 / max(len(a), 1)) + (sb.std**2 / max(len(b), 1)))
        t = (sa.mean - sb.mean) / pooled if pooled > 0 else 0.0
        return {"t_stat": float(t), "p_value": float("nan")}

