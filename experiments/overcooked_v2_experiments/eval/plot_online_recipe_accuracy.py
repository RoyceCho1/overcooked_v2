import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it in your runtime "
        "(e.g., pip install matplotlib)."
    ) from exc


def _load_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def _int(row: dict, key: str) -> int:
    return int(float(row[key]))


def _timestep_curves(rows: List[dict]):
    timesteps = sorted({_int(r, "timestep") for r in rows})
    acc = []
    coverage = []
    effective = []
    confidence = []

    for t in timesteps:
        alive = [r for r in rows if _int(r, "timestep") == t and _int(r, "episode_alive") == 1]
        valid = [r for r in alive if _int(r, "valid") == 1]
        correct = [r for r in valid if _int(r, "correct") == 1]
        prob_true = [_float(r, "prob_true_recipe") for r in valid]
        prob_true = [x for x in prob_true if np.isfinite(x)]

        acc.append(len(correct) / len(valid) if valid else np.nan)
        coverage.append(len(valid) / len(alive) if alive else np.nan)
        effective.append(len(correct) / len(alive) if alive else np.nan)
        confidence.append(float(np.mean(prob_true)) if prob_true else np.nan)

    return np.array(timesteps), np.array(acc), np.array(coverage), np.array(effective), np.array(confidence)


def _plot_curve(x, y, title: str, ylabel: str, out_path: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _pair_accuracy(rows: List[dict]) -> Tuple[np.ndarray, List[int], List[int]]:
    fcp_ids = sorted({_int(r, "fcp_run") for r in rows})
    partner_ids = sorted({_int(r, "partner_run") for r in rows})
    fcp_to_i = {x: i for i, x in enumerate(fcp_ids)}
    partner_to_j = {x: j for j, x in enumerate(partner_ids)}

    mat = np.full((len(fcp_ids), len(partner_ids)), np.nan, dtype=np.float32)
    for fcp in fcp_ids:
        for partner in partner_ids:
            group = [
                r
                for r in rows
                if _int(r, "fcp_run") == fcp and _int(r, "partner_run") == partner
            ]
            vals = [_float(r, "mean_valid_accuracy") for r in group]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                mat[fcp_to_i[fcp], partner_to_j[partner]] = float(np.mean(vals))
    return mat, fcp_ids, partner_ids


def _plot_heatmap(mat: np.ndarray, fcp_ids: List[int], partner_ids: List[int], out_path: Path, dpi: int):
    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(mat), vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    ax.set_title("Pair Mean Valid Accuracy")
    ax.set_xlabel("SP Partner Run ID")
    ax.set_ylabel("FCP Run ID")
    ax.set_xticks(np.arange(len(partner_ids)))
    ax.set_xticklabels([str(x) for x in partner_ids])
    ax.set_yticks(np.arange(len(fcp_ids)))
    ax.set_yticklabels([str(x) for x in fcp_ids])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_recipe_bar(rows: List[dict], out_path: Path, dpi: int):
    recipes = sorted({_int(r, "true_recipe") for r in rows})
    values = []
    for recipe in recipes:
        valid = [r for r in rows if _int(r, "true_recipe") == recipe and _int(r, "valid") == 1 and _int(r, "episode_alive") == 1]
        correct = [r for r in valid if _int(r, "correct") == 1]
        values.append(len(correct) / len(valid) if valid else np.nan)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar([str(r) for r in recipes], values)
    ax.set_title("Recipe-Conditioned Valid Accuracy")
    ax.set_xlabel("Recipe Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_accuracy_vs_return(rows: List[dict], out_path: Path, dpi: int):
    x = np.array([_float(r, "mean_valid_accuracy") for r in rows], dtype=np.float32)
    y = np.array([_float(r, "episode_return") for r in rows], dtype=np.float32)
    mask = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(x[mask], y[mask], s=20, alpha=0.65)
    ax.set_title("Episode Accuracy vs Return")
    ax.set_xlabel("Episode Mean Valid Accuracy")
    ax.set_ylabel("Episode Return")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep_csv", required=True, type=Path)
    parser.add_argument("--episode_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    timestep_rows = _load_csv(args.timestep_csv)
    episode_rows = _load_csv(args.episode_csv)

    x, acc, coverage, effective, confidence = _timestep_curves(timestep_rows)
    _plot_curve(x, acc, "Valid Accuracy by Timestep", "Accuracy", args.out_dir / "accuracy_by_timestep.png", args.dpi)
    _plot_curve(x, coverage, "Context Coverage by Timestep", "Coverage", args.out_dir / "coverage_by_timestep.png", args.dpi)
    _plot_curve(x, effective, "Valid and Correct Rate by Timestep", "Rate", args.out_dir / "valid_and_correct_by_timestep.png", args.dpi)
    _plot_curve(x, confidence, "Mean True-Recipe Probability by Timestep", "Probability", args.out_dir / "confidence_by_timestep.png", args.dpi)

    mat, fcp_ids, partner_ids = _pair_accuracy(episode_rows)
    _plot_heatmap(mat, fcp_ids, partner_ids, args.out_dir / "pair_accuracy_heatmap.png", args.dpi)
    _plot_recipe_bar(timestep_rows, args.out_dir / "recipe_accuracy_bar.png", args.dpi)
    _plot_accuracy_vs_return(episode_rows, args.out_dir / "episode_accuracy_vs_return.png", args.dpi)

    print("[Plot] Saved online recipe accuracy plots to:")
    print(args.out_dir.resolve())


if __name__ == "__main__":
    main()
