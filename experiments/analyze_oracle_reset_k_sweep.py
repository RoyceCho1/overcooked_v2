#!/usr/bin/env python3
import csv
from pathlib import Path


def _read_global_mean(path: Path):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == "global_mean":
                return float(row["value"])
    return None


def _read_partner_means(path: Path):
    vals = {}
    if not path.exists():
        return vals
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == "partner_mean":
                vals[int(row["partner_run"])] = float(row["value"])
    return vals


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--k_list", type=str, default="2 3 4 6")
    parser.add_argument(
        "--oracle_summary",
        type=Path,
        default=None,
        help="Optional full-oracle summary CSV. Defaults to eval_dir/fcp_oracle_summary.csv.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional output CSV. Defaults to eval_dir/oracle_reset_k_sweep_summary.csv.",
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    oracle_summary = args.oracle_summary or (eval_dir / "fcp_oracle_summary.csv")
    output_csv = args.output_csv or (eval_dir / "oracle_reset_k_sweep_summary.csv")
    k_values = [int(x) for x in args.k_list.split() if x.strip()]

    oracle_mean = _read_global_mean(oracle_summary)

    rows = []
    previous_mean = None
    for k in k_values:
        summary_path = eval_dir / f"fcp_oracle_reset_k{k}_summary.csv"
        mean = _read_global_mean(summary_path)
        if mean is None:
            rows.append(
                {
                    "k": k,
                    "global_mean": "",
                    "delta_vs_oracle": "",
                    "delta_vs_k6": "",
                    "gain_vs_previous_larger_k": "",
                    "summary_path": str(summary_path),
                }
            )
            continue
        rows.append(
            {
                "k": k,
                "global_mean": mean,
                "delta_vs_oracle": "" if oracle_mean is None else mean - oracle_mean,
                "delta_vs_k6": "",
                "gain_vs_previous_larger_k": "",
                "summary_path": str(summary_path),
            }
        )

    k6_mean = next((r["global_mean"] for r in rows if r["k"] == 6 and r["global_mean"] != ""), None)
    rows_by_desc_k = sorted([r for r in rows if r["global_mean"] != ""], key=lambda r: r["k"], reverse=True)
    previous_mean = None
    for r in rows_by_desc_k:
        if k6_mean is not None:
            r["delta_vs_k6"] = float(r["global_mean"]) - float(k6_mean)
        if previous_mean is not None:
            r["gain_vs_previous_larger_k"] = float(r["global_mean"]) - float(previous_mean)
        previous_mean = r["global_mean"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "k",
            "global_mean",
            "delta_vs_oracle",
            "delta_vs_k6",
            "gain_vs_previous_larger_k",
            "summary_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["k"]))

    print(f"oracle_mean: {oracle_mean}")
    print(f"saved: {output_csv}")
    print("k,global_mean,delta_vs_oracle,delta_vs_k6,gain_vs_previous_larger_k")
    for r in sorted(rows, key=lambda r: r["k"]):
        print(
            f"{r['k']},{r['global_mean']},{r['delta_vs_oracle']},"
            f"{r['delta_vs_k6']},{r['gain_vs_previous_larger_k']}"
        )


if __name__ == "__main__":
    main()
