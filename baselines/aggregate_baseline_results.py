import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Aggregate baseline run summaries into report-ready tables.")
    parser.add_argument("--baseline-dir", type=str, default="baselines")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    results_csv = os.path.join(args.baseline_dir, "results", "baseline_results.csv")
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"No results table found: {results_csv}")

    df = pd.read_csv(results_csv)

    # Per-seed table: useful for debugging and reproducibility.
    per_seed_cols = [
        "algorithm", "task", "reward_type", "seed", "episodes",
        "final_eval_success", "best_eval_success", "last5_eval_success_mean",
        "final_eval_return", "last5_eval_return_mean", "model_path", "csv_path",
    ]
    per_seed = df[[c for c in per_seed_cols if c in df.columns]].copy()

    # Report table: mean/std across seeds.
    grouped = (
        df.groupby(["algorithm", "task", "reward_type"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            success_mean=("last5_eval_success_mean", "mean"),
            success_std=("last5_eval_success_mean", "std"),
            best_success_mean=("best_eval_success", "mean"),
            return_mean=("last5_eval_return_mean", "mean"),
            return_std=("last5_eval_return_mean", "std"),
        )
    )

    grouped["success_percent_mean"] = 100.0 * grouped["success_mean"]
    grouped["success_percent_std"] = 100.0 * grouped["success_std"]
    grouped["best_success_percent_mean"] = 100.0 * grouped["best_success_mean"]

    out_dir = args.out or os.path.join(args.baseline_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    per_seed_path = os.path.join(out_dir, "baseline_per_seed_table.csv")
    report_path = os.path.join(out_dir, "baseline_report_table.csv")
    latex_path = os.path.join(out_dir, "baseline_report_table.tex")

    per_seed.to_csv(per_seed_path, index=False)
    grouped.to_csv(report_path, index=False)

    latex_cols = [
        "algorithm", "task", "reward_type", "seeds",
        "success_percent_mean", "success_percent_std",
        "best_success_percent_mean", "return_mean",
    ]
    latex_df = grouped[latex_cols].copy()
    latex_df["success_percent_mean"] = latex_df["success_percent_mean"].round(2)
    latex_df["success_percent_std"] = latex_df["success_percent_std"].round(2)
    latex_df["best_success_percent_mean"] = latex_df["best_success_percent_mean"].round(2)
    latex_df["return_mean"] = latex_df["return_mean"].round(2)

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_df.to_latex(index=False, escape=False))

    print(f"Saved per-seed table: {per_seed_path}")
    print(f"Saved report table:   {report_path}")
    print(f"Saved LaTeX table:    {latex_path}")


if __name__ == "__main__":
    main()
