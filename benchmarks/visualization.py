from __future__ import annotations

from pathlib import Path


def plot_metric_comparison(
    metric_to_config_values: dict[str, dict[str, float]],
    out_path: str | Path,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    metrics = list(metric_to_config_values.keys())
    configs = sorted({c for d in metric_to_config_values.values() for c in d.keys()})
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, max(3, 2.5 * len(metrics))))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        vals = [metric_to_config_values[m].get(c, 0.0) for c in configs]
        ax.bar(configs, vals)
        ax.set_title(m)
    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=250)
    plt.close(fig)
    return str(p)


def plot_curve(y_values: list[float], title: str, out_path: str | Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(range(1, len(y_values) + 1)), y_values)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    fig.tight_layout()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=250)
    plt.close(fig)
    return str(p)

