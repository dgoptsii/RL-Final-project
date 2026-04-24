from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def find_x_column(df: pd.DataFrame) -> str:
    for col in ["episode", "total_steps", "step"]:
        if col in df.columns:
            return col
    return "__index__"


def choose_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def outdir_for_csv(base_outdir: str, csv_path: str) -> str:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join(base_outdir, stem)


def plot_metric(df: pd.DataFrame, x_col: str, y_col: str, out_path: str, title: str, ylabel: str, smooth: int = 10, sparse: bool = False):
    y = df[y_col]
    x = df[x_col] if x_col != "__index__" else pd.Series(range(len(df)))
    mask = y.notna()
    x_valid = x[mask].reset_index(drop=True)
    y_valid = y[mask].reset_index(drop=True)
    if len(y_valid) == 0:
        print(f"Skipping {y_col}: no values")
        return
    plt.figure(figsize=(8, 5))
    if sparse:
        plt.plot(x_valid, y_valid, marker="o", linestyle="-", alpha=0.5, label="raw")
    else:
        plt.plot(x_valid, y_valid, alpha=0.35, label="raw")
    plt.plot(x_valid, moving_average(y_valid, smooth), linewidth=2, label=f"MA{smooth}")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dual(df: pd.DataFrame, x_col: str, col_a: str, col_b: str, out_path: str, title: str, ylabel: str, smooth: int):
    plt.figure(figsize=(8, 5))
    x = df[x_col] if x_col != "__index__" else pd.Series(range(len(df)))
    plotted = False
    for col, label in [(col_a, col_a), (col_b, col_b)]:
        if col not in df.columns:
            continue
        y = df[col]
        mask = y.notna()
        x_valid = x[mask].reset_index(drop=True)
        y_valid = y[mask].reset_index(drop=True)
        if len(y_valid) == 0:
            continue
        plt.plot(x_valid, moving_average(y_valid, smooth), linewidth=2, label=label)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_from_csv(csv_path: str, outdir: str = "plots_hiro_her", smooth: int = 10) -> None:
    df = pd.read_csv(csv_path)
    df["__index__"] = range(len(df))
    x_col = find_x_column(df)
    run_outdir = outdir_for_csv(outdir, csv_path)
    os.makedirs(run_outdir, exist_ok=True)

    metrics = [
        ("train_return", "Training Return", "Return", False),
        ("eval_return", "Evaluation Return", "Return", True),
        ("eval_success", "Evaluation Success Rate", "Success Rate", True),
        ("train_true_success", "Training True Goal Success", "Success", False),
        ("train_subgoal_success", "Training Subgoal Success", "Success", False),
        ("eval_subgoal_success", "Evaluation Subgoal Success", "Success", True),
        ("train_true_goal_distance", "Training True Goal Distance", "Distance", False),
        ("train_subgoal_distance", "Training Subgoal Distance", "Distance", False),
        ("eval_true_goal_distance", "Evaluation True Goal Distance", "Distance", True),
        ("eval_subgoal_distance", "Evaluation Subgoal Distance", "Distance", True),
        ("critic_loss", "Low-level Critic Loss", "Loss", False),
        ("actor_loss", "Low-level Actor Loss", "Loss", False),
        ("q_mean", "Low-level Mean Q", "Q", False),
        ("high_critic_loss", "High-level Critic Loss", "Loss", False),
        ("high_actor_loss", "High-level Actor Loss", "Loss", False),
        ("high_q_mean", "High-level Mean Q", "Q", False),
        ("her_transitions_added", "HER Transitions Added", "Count", False),
        ("replay_size", "Replay Buffer Size", "Transitions", False),
    ]

    for col, title, ylabel, sparse in metrics:
        if col in df.columns:
            plot_metric(df, x_col, col, os.path.join(run_outdir, f"{col}.png"), title, ylabel, smooth=smooth, sparse=sparse)

    if "train_true_success" in df.columns and "train_subgoal_success" in df.columns:
        plot_dual(df, x_col, "train_true_success", "train_subgoal_success", os.path.join(run_outdir, "train_true_vs_subgoal_success.png"), "Training True vs Subgoal Success", "Success", smooth)
    if "train_true_goal_distance" in df.columns and "train_subgoal_distance" in df.columns:
        plot_dual(df, x_col, "train_true_goal_distance", "train_subgoal_distance", os.path.join(run_outdir, "train_true_vs_subgoal_distance.png"), "Training True vs Subgoal Distance", "Distance", smooth)

    print(f"Plots saved to: {run_outdir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--outdir", type=str, default="plots_hiro_her")
    parser.add_argument("--smooth", type=int, default=10)
    args = parser.parse_args()
    for csv_path in args.csv:
        plot_from_csv(csv_path, args.outdir, smooth=args.smooth)


if __name__ == "__main__":
    main()
