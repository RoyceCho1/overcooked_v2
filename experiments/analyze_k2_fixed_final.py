#!/usr/bin/env python3
import csv
import math
from pathlib import Path


def _global_mean(path: Path):
    if not path.exists():
        return math.nan
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == "global_mean":
                return float(row["value"])
    return math.nan


def _partner_means(path: Path):
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
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    eval_dir = args.eval_dir
    output_csv = args.output_csv or (eval_dir / "k2_fixed_final_comparison.csv")

    summaries = {
        "base": eval_dir / "fcp_base_summary.csv",
        "encoder2_k6": eval_dir / "fcp_encoder2_summary.csv",
        "encoder_k2_fixed": eval_dir / "fcp_encoder_k2_fixed_summary.csv",
        "oracle_reset_k2": eval_dir / "fcp_oracle_reset_k2_summary.csv",
        "oracle": eval_dir / "fcp_oracle_summary.csv",
    }
    means = {name: _global_mean(path) for name, path in summaries.items()}

    base = means["base"]
    k2 = means["encoder_k2_fixed"]
    oracle_reset_k2 = means["oracle_reset_k2"]
    oracle = means["oracle"]

    rows = []
    for name, value in means.items():
        rows.append({"section": "global", "metric": name, "value": value})

    derived = {
        "encoder_k2_minus_base": k2 - base,
        "encoder_k2_minus_encoder2_k6": k2 - means["encoder2_k6"],
        "oracle_reset_k2_minus_encoder_k2": oracle_reset_k2 - k2,
        "oracle_minus_oracle_reset_k2": oracle - oracle_reset_k2,
        "encoder_k2_utilization_vs_oracle_reset_k2": (
            (k2 - base) / (oracle_reset_k2 - base)
            if math.isfinite(k2) and math.isfinite(base) and oracle_reset_k2 > base
            else math.nan
        ),
    }
    for name, value in derived.items():
        rows.append({"section": "derived", "metric": name, "value": value})

    partner_tables = {name: _partner_means(path) for name, path in summaries.items()}
    all_partners = sorted(set().union(*[set(v.keys()) for v in partner_tables.values()]))
    for partner in all_partners:
        p_base = partner_tables["base"].get(partner, math.nan)
        p_k2 = partner_tables["encoder_k2_fixed"].get(partner, math.nan)
        p_k6 = partner_tables["encoder2_k6"].get(partner, math.nan)
        p_oracle_reset = partner_tables["oracle_reset_k2"].get(partner, math.nan)
        rows.append(
            {
                "section": "partner",
                "metric": f"partner_{partner}_encoder_k2_minus_base",
                "value": p_k2 - p_base,
            }
        )
        rows.append(
            {
                "section": "partner",
                "metric": f"partner_{partner}_encoder_k2_minus_encoder2_k6",
                "value": p_k2 - p_k6,
            }
        )
        rows.append(
            {
                "section": "partner",
                "metric": f"partner_{partner}_oracle_reset_k2_minus_encoder_k2",
                "value": p_oracle_reset - p_k2,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {output_csv}")
    print("Global means:")
    for name, value in means.items():
        print(f"  {name}: {value:.6f}" if math.isfinite(value) else f"  {name}: missing")
    print("Derived:")
    for name, value in derived.items():
        print(f"  {name}: {value:.6f}" if math.isfinite(value) else f"  {name}: missing")


if __name__ == "__main__":
    main()
