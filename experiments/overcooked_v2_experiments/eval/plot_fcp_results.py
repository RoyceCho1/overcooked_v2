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


def _load_detail_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "variant": r["variant"],
                    "pair_order": int(float(r["pair_order"])),
                    "fcp_run": int(float(r["fcp_run"])),
                    "partner_run": int(float(r["partner_run"])),
                    "mean_return": float(r["mean_return"]),
                    "std_return": float(r["std_return"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows in CSV: {path}")
    return rows


def _collect_ids(*rows_groups: List[dict]) -> Tuple[List[int], List[int]]:
    fcp_ids = sorted(
        {r["fcp_run"] for rows in rows_groups for r in rows}
    )
    partner_ids = sorted(
        {r["partner_run"] for rows in rows_groups for r in rows}
    )
    return fcp_ids, partner_ids


def _build_matrix(rows: List[dict], fcp_ids: List[int], partner_ids: List[int]) -> np.ndarray:
    fcp_to_i = {rid: i for i, rid in enumerate(fcp_ids)}
    partner_to_j = {rid: j for j, rid in enumerate(partner_ids)}

    mat = np.full((len(fcp_ids), len(partner_ids)), np.nan, dtype=np.float32)
    for r in rows:
        i = fcp_to_i[r["fcp_run"]]
        j = partner_to_j[r["partner_run"]]
        mat[i, j] = r["mean_return"]
    return mat


def _plot_partner_bar(
    base_partner_mean: np.ndarray,
    enc_partner_mean: np.ndarray,
    oracle_partner_mean: np.ndarray,
    partner_ids: List[int],
    out_path: Path,
    dpi: int,
):
    x = np.arange(len(partner_ids))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(x - width, base_partner_mean, width=width, label="base")
    ax.bar(x, enc_partner_mean, width=width, label="encoder")
    ax.bar(x + width, oracle_partner_mean, width=width, label="oracle")

    ax.set_title("Partner-Wise Mean Return")
    ax.set_xlabel("SP Partner Run ID")
    ax.set_ylabel("Mean Return")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in partner_ids])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_delta_heatmap(
    enc_delta: np.ndarray,
    oracle_delta: np.ndarray,
    fcp_ids: List[int],
    partner_ids: List[int],
    out_path: Path,
    dpi: int,
):
    stacked = np.concatenate(
        [enc_delta.reshape(-1), oracle_delta.reshape(-1)]
    )
    finite = stacked[np.isfinite(stacked)]
    vmax = float(np.max(np.abs(finite))) if finite.size > 0 else 1.0

    cmap = plt.cm.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#d9d9d9")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    m0 = np.ma.masked_invalid(enc_delta)
    im0 = axes[0].imshow(m0, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_title("Delta Heatmap: Encoder - Base")
    axes[0].set_xlabel("SP Partner Run ID")
    axes[0].set_ylabel("FCP Run ID")
    axes[0].set_xticks(np.arange(len(partner_ids)))
    axes[0].set_xticklabels([str(x) for x in partner_ids], rotation=0)
    axes[0].set_yticks(np.arange(len(fcp_ids)))
    axes[0].set_yticklabels([str(x) for x in fcp_ids])

    m1 = np.ma.masked_invalid(oracle_delta)
    im1 = axes[1].imshow(m1, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_title("Delta Heatmap: Oracle - Base")
    axes[1].set_xlabel("SP Partner Run ID")
    axes[1].set_ylabel("FCP Run ID")
    axes[1].set_xticks(np.arange(len(partner_ids)))
    axes[1].set_xticklabels([str(x) for x in partner_ids], rotation=0)
    axes[1].set_yticks(np.arange(len(fcp_ids)))
    axes[1].set_yticklabels([str(x) for x in fcp_ids])

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Return Delta")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _global_mean(rows: List[dict]) -> float:
    return float(np.mean([r["mean_return"] for r in rows]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_detail",
        type=Path,
        default=Path("/home/myuser/overcooked_v2_experiments/runs/eval/fcp_base_detail.csv"),
    )
    parser.add_argument(
        "--encoder_detail",
        type=Path,
        default=Path("/home/myuser/overcooked_v2_experiments/runs/eval/fcp_encoder_detail.csv"),
    )
    parser.add_argument(
        "--oracle_detail",
        type=Path,
        default=Path("/home/myuser/overcooked_v2_experiments/runs/eval/fcp_oracle_detail.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/myuser/overcooked_v2_experiments/runs/eval/plots"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    base_rows = _load_detail_csv(args.base_detail)
    enc_rows = _load_detail_csv(args.encoder_detail)
    oracle_rows = _load_detail_csv(args.oracle_detail)

    fcp_ids, partner_ids = _collect_ids(base_rows, enc_rows, oracle_rows)
    base_mat = _build_matrix(base_rows, fcp_ids, partner_ids)
    enc_mat = _build_matrix(enc_rows, fcp_ids, partner_ids)
    oracle_mat = _build_matrix(oracle_rows, fcp_ids, partner_ids)

    base_partner_mean = np.nanmean(base_mat, axis=0)
    enc_partner_mean = np.nanmean(enc_mat, axis=0)
    oracle_partner_mean = np.nanmean(oracle_mat, axis=0)

    enc_delta = enc_mat - base_mat
    oracle_delta = oracle_mat - base_mat

    partner_bar_path = args.out_dir / "partner_mean_bar.png"
    delta_heatmap_path = args.out_dir / "delta_heatmap.png"

    _plot_partner_bar(
        base_partner_mean=base_partner_mean,
        enc_partner_mean=enc_partner_mean,
        oracle_partner_mean=oracle_partner_mean,
        partner_ids=partner_ids,
        out_path=partner_bar_path,
        dpi=args.dpi,
    )
    _plot_delta_heatmap(
        enc_delta=enc_delta,
        oracle_delta=oracle_delta,
        fcp_ids=fcp_ids,
        partner_ids=partner_ids,
        out_path=delta_heatmap_path,
        dpi=args.dpi,
    )

    base_g = _global_mean(base_rows)
    enc_g = _global_mean(enc_rows)
    oracle_g = _global_mean(oracle_rows)

    print("[Plot] Saved:")
    print(f"- {partner_bar_path.resolve()}")
    print(f"- {delta_heatmap_path.resolve()}")
    print("[Global Mean]")
    print(f"- base:    {base_g:.3f}")
    print(f"- encoder: {enc_g:.3f} (delta vs base: {enc_g - base_g:+.3f})")
    print(f"- oracle:  {oracle_g:.3f} (delta vs base: {oracle_g - base_g:+.3f})")


if __name__ == "__main__":
    main()
