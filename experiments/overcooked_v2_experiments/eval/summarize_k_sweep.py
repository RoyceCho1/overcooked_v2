import argparse
import csv
import re
from pathlib import Path


def _extract_k(path: Path) -> int:
    m = re.search(r"k(\d+)", str(path))
    if not m:
        return -1
    return int(m.group(1))


def _read_global_mean(summary_csv: Path):
    if not summary_csv.exists():
        return None
    with open(summary_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("metric") == "global_mean":
                return float(row["value"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_root",
        type=Path,
        default=Path("/home/myuser/overcooked_v2_experiments/runs/k_sweep_demo_cook_simple"),
    )
    args = parser.parse_args()

    if not args.sweep_root.exists():
        raise FileNotFoundError(f"sweep_root not found: {args.sweep_root}")

    rows = []
    for k_dir in sorted([p for p in args.sweep_root.iterdir() if p.is_dir()]):
        k = _extract_k(k_dir)
        if k < 0:
            continue

        summary_csv = k_dir / f"fcp_encoder_summary_k{k}.csv"
        detail_csv = k_dir / f"fcp_encoder_detail_k{k}.csv"
        gm = _read_global_mean(summary_csv)
        if gm is None:
            continue

        n_rows = 0
        if detail_csv.exists():
            with open(detail_csv, "r", encoding="utf-8", newline="") as f:
                n_rows = max(sum(1 for _ in f) - 1, 0)

        rows.append((k, gm, n_rows, str(summary_csv)))

    rows.sort(key=lambda x: x[0])
    if not rows:
        print("No completed K results found.")
        return

    print("K-sweep encoder results (higher global_mean is better):")
    print("K\tglobal_mean\tn_pairs\tsummary_csv")
    for k, gm, n_rows, path in rows:
        print(f"{k}\t{gm:.3f}\t{n_rows}\t{path}")

    best = max(rows, key=lambda x: x[1])
    print("")
    print(f"Best K: {best[0]} (global_mean={best[1]:.3f})")


if __name__ == "__main__":
    main()
