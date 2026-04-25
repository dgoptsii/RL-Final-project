from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def find_x_column(df: pd.DataFrame) -> str:
    for col in ["episode", "episodes", "Episode", "ep", "total_steps", "step", "steps", "update", "updates"]:
        if col in df.columns:
            return col
    return "__index__"


def choose_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_x_values(df: pd.DataFrame, x_col: str) -> pd.Series:
    if x_col == "__index__":
        return pd.Series(range(len(df)))
    return df[x_col]


def outdir_for_csv(base_outdir: str, csv_path: str) -> str:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join(base_outdir, stem)


def _safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


METRICS: Dict[str, Tuple[str, str, bool]] = {
    "train_return": ("Training Return", "Return", False),
    "return": ("Training Return", "Return", False),
    "episode_return": ("Episode Return", "Return", False),
    "eval_return": ("Evaluation Return", "Return", True),
    "train_success": ("Training Success", "Success", False),
    "train_success_rate": ("Training Success", "Success", False),
    "train_curriculum_success": ("Training Curriculum Goal Success", "Success", False),
    "train_true_success": ("Training True Goal Success", "Success", False),
    "eval_success": ("Evaluation Success", "Success", True),
    "success_rate": ("Evaluation Success", "Success", True),
    "eval_success_rate": ("Evaluation Success", "Success", True),
    "train_true_goal_distance": ("Training True Goal Distance", "Distance", False),
    "train_curriculum_goal_distance": ("Training Curriculum Goal Distance", "Distance", False),
    "eval_true_goal_distance": ("Evaluation True Goal Distance", "Distance", True),
    "curriculum_ratio": ("Curriculum Ratio", "Ratio", False),
    "goal_intermediate_frac": ("GoalGAN Intermediate Goal Fraction", "Fraction", False),
    "goal_success_mean": ("GoalGAN Goal Success Mean", "Success", False),
    "goal_buffer_size": ("GoalGAN Buffer Size", "Goals", False),
    "critic_loss": ("Critic Loss", "Loss", False),
    "actor_loss": ("Actor Loss", "Loss", False),
    "q_mean": ("Mean Q", "Q", False),
    "alpha": ("SAC Entropy Temperature", "Alpha", False),
    "td_error_mean": ("Mean TD Error", "TD error", False),
    "td_error_max": ("Max TD Error", "TD error", False),
    "importance_weight_mean": ("Mean Importance Weight", "Weight", False),
    "priority_mean": ("Replay Priority Mean", "Priority", False),
    "priority_max": ("Replay Priority Max", "Priority", False),
    "her_transitions_added": ("HER Transitions Added", "Count", False),
    "replay_size": ("Replay Buffer Size", "Transitions", False),
}


def plot_metric(df: pd.DataFrame, x_col: str, y_col: str, out_path: str, title: Optional[str] = None,
                ylabel: Optional[str] = None, smooth: int = 10, sparse: Optional[bool] = None) -> None:
    if y_col not in df.columns:
        return
    y = df[y_col]
    x = get_x_values(df, x_col)
    mask = y.notna()
    x_valid = x[mask].reset_index(drop=True)
    y_valid = y[mask].reset_index(drop=True)
    if len(y_valid) == 0:
        print(f"Skipping {y_col}: no non-NaN values")
        return

    default_title, default_ylabel, default_sparse = METRICS.get(y_col, (y_col, y_col, False))
    title = title or default_title
    ylabel = ylabel or default_ylabel
    sparse = default_sparse if sparse is None else sparse

    plt.figure(figsize=(8, 5))
    if sparse:
        plt.plot(x_valid, y_valid, marker="o", linestyle="-", alpha=0.45, label="raw")
    else:
        plt.plot(x_valid, y_valid, alpha=0.30, label="raw")
    if smooth > 1:
        plt.plot(x_valid, moving_average(y_valid, smooth), linewidth=2, label=f"MA{smooth}")
    plt.title(title)
    plt.xlabel("Episode" if x_col == "__index__" else x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_dual(df: pd.DataFrame, x_col: str, col_a: str, col_b: str, out_path: str,
              title: str, ylabel: str, smooth: int = 10) -> None:
    plt.figure(figsize=(8, 5))
    x = get_x_values(df, x_col)
    plotted = False
    for col in [col_a, col_b]:
        if col not in df.columns:
            continue
        y = df[col]
        mask = y.notna()
        x_valid = x[mask].reset_index(drop=True)
        y_valid = y[mask].reset_index(drop=True)
        if len(y_valid) == 0:
            continue
        plt.plot(x_valid, y_valid, alpha=0.20)
        plt.plot(x_valid, moving_average(y_valid, smooth), linewidth=2, label=col)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(title)
    plt.xlabel("Episode" if x_col == "__index__" else x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_single_run(df: pd.DataFrame, x_col: str, outdir: str, smooth: int = 10) -> None:
    os.makedirs(outdir, exist_ok=True)
    for col, (title, ylabel, sparse) in METRICS.items():
        if col in df.columns:
            plot_metric(df, x_col, col, os.path.join(outdir, f"{col}.png"), title, ylabel, smooth, sparse)

    if "train_true_success" in df.columns and "train_curriculum_success" in df.columns:
        plot_dual(df, x_col, "train_true_success", "train_curriculum_success",
                  os.path.join(outdir, "train_true_vs_curriculum_success.png"),
                  "Training Success: True Goal vs Curriculum Goal", "Success", smooth)
    if "train_true_goal_distance" in df.columns and "train_curriculum_goal_distance" in df.columns:
        plot_dual(df, x_col, "train_true_goal_distance", "train_curriculum_goal_distance",
                  os.path.join(outdir, "train_true_vs_curriculum_distance.png"),
                  "Training Distance: True Goal vs Curriculum Goal", "Distance", smooth)


def plot_from_csv(csv_path: str, outdir: str = "results/plots", smooth: int = 10) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")
    df["__index__"] = range(len(df))
    x_col = find_x_column(df)
    run_outdir = outdir_for_csv(outdir, csv_path)
    plot_single_run(df, x_col, run_outdir, smooth)
    print(f"Plots saved to: {run_outdir}/")


def make_comparison_plot(dfs: List[pd.DataFrame], labels: List[str], x_col: str, y_col: str,
                         title: str, ylabel: str, save_path: str, smooth: int = 10,
                         sparse: bool = False) -> None:
    plt.figure(figsize=(9, 5.5))
    plotted_any = False
    for df, label in zip(dfs, labels):
        if y_col not in df.columns:
            continue
        x = get_x_values(df, x_col)
        y = df[y_col]
        mask = y.notna()
        x_valid = x[mask].reset_index(drop=True)
        y_valid = y[mask].reset_index(drop=True)
        if len(y_valid) == 0:
            continue
        if sparse:
            plt.plot(x_valid, y_valid, marker="o", linestyle="-", alpha=0.25)
        plt.plot(x_valid, moving_average(y_valid, smooth), linewidth=2, label=label)
        plotted_any = True
    if not plotted_any:
        print(f"Skipping comparison {y_col}: no valid values")
        plt.close()
        return
    plt.title(title)
    plt.xlabel("Episode" if x_col == "__index__" else x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_comparison(csv_paths: List[str], labels: Optional[List[str]], outdir: str, smooth: int = 10) -> None:
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if df.empty:
            raise ValueError(f"Empty CSV: {p}")
        df["__index__"] = range(len(df))
        dfs.append(df)
    if labels is None:
        labels = [Path(p).stem for p in csv_paths]
    if len(labels) != len(csv_paths):
        raise ValueError("Number of labels must match number of CSV paths")
    x_col = find_x_column(dfs[0])
    os.makedirs(outdir, exist_ok=True)

    preferred = [
        "eval_success", "train_true_success", "train_curriculum_success",
        "eval_true_goal_distance", "train_true_goal_distance", "train_curriculum_goal_distance",
        "train_return", "eval_return", "curriculum_ratio", "goal_intermediate_frac",
        "critic_loss", "actor_loss", "q_mean", "td_error_mean", "priority_mean",
    ]
    for col in preferred:
        if not any(col in df.columns for df in dfs):
            continue
        title, ylabel, sparse = METRICS.get(col, (col, col, False))
        make_comparison_plot(dfs, labels, x_col, col, f"Comparison: {title}", ylabel,
                             os.path.join(outdir, f"compare_{col}.png"), smooth, sparse)


def main():
    parser = argparse.ArgumentParser(description="Unified plotter for baseline, HER, prioritized HER, curriculum, and GoalGAN runs.")
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--outdir", type=str, default="results/plots")
    parser.add_argument("--smooth", type=int, default=10)
    args = parser.parse_args()

    if len(args.csv) == 1:
        plot_from_csv(args.csv[0], outdir=args.outdir, smooth=args.smooth)
    else:
        plot_comparison(args.csv, args.labels, os.path.join(args.outdir, "compare"), args.smooth)
        for p in args.csv:
            plot_from_csv(p, outdir=args.outdir, smooth=args.smooth)


if __name__ == "__main__":
    main()
