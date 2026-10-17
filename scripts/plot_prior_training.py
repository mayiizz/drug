from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_prior_curves(metrics_path: Path, out_path: Path) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = payload.get("history", []) or []
    if not history:
        raise ValueError(f"No history found in {metrics_path}")

    steps = [int(h.get("step", 0)) for h in history]
    losses = [float(h.get("loss", 0.0)) for h in history]
    raw_valid = [float(h.get("raw_valid_fraction", 0.0)) for h in history]
    unique = [float(h.get("unique_fraction", 0.0)) for h in history]
    novelty = [float(h.get("novelty_fraction", 0.0)) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes.ravel()
    ax[0].plot(steps, losses)
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Loss")

    ax[1].plot(steps, raw_valid)
    ax[1].set_title("Raw Valid Fraction")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Fraction")
    ax[1].set_ylim(0.0, 1.0)

    ax[2].plot(steps, unique)
    ax[2].set_title("Uniqueness")
    ax[2].set_xlabel("Step")
    ax[2].set_ylabel("Fraction")
    ax[2].set_ylim(0.0, 1.0)

    ax[3].plot(steps, novelty)
    ax[3].set_title("Novelty")
    ax[3].set_xlabel("Step")
    ax[3].set_ylabel("Fraction")
    ax[3].set_ylim(0.0, 1.0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot prior training curves")
    p.add_argument("--metrics", type=Path, default=Path("results/prior_training_metrics.json"))
    p.add_argument("--out", type=Path, default=Path("results/prior_training_curves.png"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plot_prior_curves(args.metrics, args.out)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()

