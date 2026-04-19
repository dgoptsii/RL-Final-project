import argparse
import os
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_x_column(df: pd.DataFrame) -> str:
    """Pick the best x-axis column. Preference: episode -> step -> update -> index."""
    for col in ["episode", "episodes", "Episode", "ep", "step", "steps", "update", "updates"]:
        if col in df.columns:
            return col
    return "__index__"


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def choose_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_x_values(df: pd.DataFrame, x_col: str) -> pd.Series:
    if x_col == "__index__":
        return pd.Series(range(len(df)))
    return df[x_col]


def outdir_for_csv(base_outdir: str, csv_path: str) -> str:
    """
    Return a per-run subdirectory under base_outdir named after the CSV stem.

    Example:
        base_outdir = "plots"
        csv_path    = "logs/td3_FetchReach_dense_seed0.csv"
        -> "plots/td3_FetchReach_dense_seed0/"
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join(base_outdir, stem)


# ---------------------------------------------------------------------------
# Single-run plots
# ---------------------------------------------------------------------------

def make_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    save_path: str,
    smooth_window: int = 10,
) -> None:
    """Standard dense plot (logged every episode). Drops NaNs."""
    plt.figure(figsize=(8, 5))

    x = get_x_values(df, x_col)
    y = df[y_col]
    mask = y.notna()
    x_valid = x[mask]
    y_valid = y[mask]

    plt.plot(x_valid, y_valid, label=y_col)
    plt.plot(
        x_valid,
        moving_average(y_valid.reset_index(drop=True), smooth_window),
        label=f"{y_col} (MA{smooth_window})",
    )

    plt.title(title)
    plt.xlabel("Episode" if x_col == "__index__" else x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def make_dual_plot(
    df: pd.DataFrame,
    x_col: str,
    col_a: str,
    col_b: str,
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    save_path: str,
    smooth_window: int = 10,
) -> None:
    """
    Plot two series on the same axes (e.g. curriculum vs true goal success).
    Each series gets its own colour; the smoothed version is a darker shade.
    """
    plt.figure(figsize=(8, 5))

    for col, label, color in [
        (col_a, label_a, "tab:blue"),
        (col_b, label_b, "tab:orange"),
    ]:
        if col not in df.columns:
            continue
        x = get_x_values(df, x_col)
        y = df[col]
        mask = y.notna()
        x_valid = x[mask]
        y_valid = y[mask].reset_index(drop=True)

        plt.plot(x_valid, y_valid, alpha=0.35, color=color)
        plt.plot(
            x_valid,
            moving_average(y_valid, smooth_window),
            label=f"{label} (MA{smooth_window})",
            color=color,
            linewidth=2,
        )

    plt.title(title)
    plt.xlabel("Episode" if x_col == "__index__" else x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def make_sparse_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    save_path: str,
    smooth_window: int = 5,
) -> None:
    """Plot for metrics logged only every N episodes (NaN otherwise)."""
    plt.figure(figsize=(8, 5))

    x = get_x_values(df, x_col)
    y = df[y_col]
    mask = y.notna()
    x_valid = x[mask].reset_index(drop=True)
    y_valid = y[mask].reset_index(drop=True)

    if len(y_valid) == 0:
        print(f"Skipping {y_col}: no non-NaN values found.")
        plt.close()
        return

    plt.plot(x_valid, y_valid, marker="o", linestyle="-", label=y_col)
    plt.plot(
        x_valid,
        moving_average(y_valid, smooth_window),
        label=f"{y_col} (MA{smooth_window})",
    )

    plt.title(title)
    plt.xlabel("Episode" if x_col == "__index__" else x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_single_run(df: pd.DataFrame, x_col: str, outdir: str, smooth: int) -> None:
    """Generate all per-run plots into outdir."""
    os.makedirs(outdir, exist_ok=True)

    return_col = choose_first_existing(df, ["return", "episode_return", "train_return", "reward", "ep_return"])
    eval_return_col = choose_first_existing(df, ["eval_return"])
    eval_success_col = choose_first_existing(df, ["eval_success", "success_rate", "eval_success_rate"])
    critic_loss_col = choose_first_existing(df, ["critic_loss", "q_loss"])
    actor_loss_col = choose_first_existing(df, ["actor_loss", "policy_loss"])
    q_mean_col = choose_first_existing(df, ["q_mean", "q_value", "critic_q_mean"])

    # Dual-success columns (new schema)
    curriculum_success_col = choose_first_existing(df, ["train_curriculum_success"])
    true_success_col = choose_first_existing(df, ["train_true_success"])
    # Legacy single-column fallback
    legacy_train_success_col = choose_first_existing(df, ["train_success", "train_success_rate"])

    if return_col:
        make_plot(df, x_col, return_col,
                  title="Episode Return", ylabel="Return",
                  save_path=os.path.join(outdir, "episode_return.png"),
                  smooth_window=smooth)

    if eval_return_col:
        make_sparse_plot(df, x_col, eval_return_col,
                         title="Evaluation Return", ylabel="Return",
                         save_path=os.path.join(outdir, "eval_return.png"),
                         smooth_window=max(3, min(smooth, 5)))

    if eval_success_col:
        make_sparse_plot(df, x_col, eval_success_col,
                         title="Evaluation Success Rate", ylabel="Success Rate",
                         save_path=os.path.join(outdir, "eval_success.png"),
                         smooth_window=max(3, min(smooth, 5)))

    # --- Training success: dual plot if both new columns are present ---
    if curriculum_success_col and true_success_col:
        # Combined dual plot
        make_dual_plot(
            df, x_col,
            col_a=curriculum_success_col, label_a="Curriculum Goal",
            col_b=true_success_col,       label_b="True Goal",
            title="Training Success Rate (Curriculum vs True Goal)",
            ylabel="Success Rate",
            save_path=os.path.join(outdir, "train_success_dual.png"),
            smooth_window=smooth,
        )
        # Also save them individually for easy reference
        make_plot(df, x_col, curriculum_success_col,
                  title="Training Success Rate (Curriculum Goal)",
                  ylabel="Success Rate",
                  save_path=os.path.join(outdir, "train_curriculum_success.png"),
                  smooth_window=smooth)
        make_plot(df, x_col, true_success_col,
                  title="Training Success Rate (True Goal)",
                  ylabel="Success Rate",
                  save_path=os.path.join(outdir, "train_true_success.png"),
                  smooth_window=smooth)
    elif legacy_train_success_col:
        # Backward-compatible single plot for old CSVs
        make_plot(df, x_col, legacy_train_success_col,
                  title="Training Success Rate", ylabel="Success Rate",
                  save_path=os.path.join(outdir, "train_success.png"),
                  smooth_window=smooth)

    if critic_loss_col:
        make_plot(df, x_col, critic_loss_col,
                  title="Critic Loss", ylabel="Loss",
                  save_path=os.path.join(outdir, "critic_loss.png"),
                  smooth_window=smooth)

    if actor_loss_col:
        make_plot(df, x_col, actor_loss_col,
                  title="Actor Loss", ylabel="Loss",
                  save_path=os.path.join(outdir, "actor_loss.png"),
                  smooth_window=smooth)

    if q_mean_col:
        make_plot(df, x_col, q_mean_col,
                  title="Mean Q Value", ylabel="Q Mean",
                  save_path=os.path.join(outdir, "q_mean.png"),
                  smooth_window=smooth)


# ---------------------------------------------------------------------------
# Multi-run comparison plots
# ---------------------------------------------------------------------------

def make_comparison_plot(
    dfs: List[pd.DataFrame],
    labels: List[str],
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    save_path: str,
    smooth_window: int = 10,
    sparse: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    plotted_any = False

    for df, label in zip(dfs, labels):
        x = get_x_values(df, x_col)
        y = df[y_col]
        mask = y.notna()
        x_valid = x[mask].reset_index(drop=True)
        y_valid = y[mask].reset_index(drop=True)

        if len(y_valid) == 0:
            continue

        if sparse:
            plt.plot(x_valid, y_valid, marker="o", linestyle="-", alpha=0.5, label=f"{label} raw")
            plt.plot(x_valid, moving_average(y_valid, smooth_window), label=f"{label} MA{smooth_window}")
        else:
            plt.plot(x_valid, moving_average(y_valid, smooth_window), label=label)

        plotted_any = True

    if not plotted_any:
        print(f"Skipping comparison plot {y_col}: no valid values found.")
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Episode" if x_col == "__index__" else x_col)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison(
    dfs: List[pd.DataFrame],
    labels: List[str],
    x_col: str,
    outdir: str,
    smooth: int,
) -> None:
    """Generate all comparison plots into outdir."""
    os.makedirs(outdir, exist_ok=True)

    df0 = dfs[0]

    def exists_in_all(col: Optional[str]) -> bool:
        return col is not None and all(col in df.columns for df in dfs)

    return_col       = choose_first_existing(df0, ["return", "episode_return", "train_return", "reward", "ep_return"])
    eval_return_col  = choose_first_existing(df0, ["eval_return"])
    eval_success_col = choose_first_existing(df0, ["eval_success", "success_rate", "eval_success_rate"])
    critic_loss_col  = choose_first_existing(df0, ["critic_loss", "q_loss"])
    actor_loss_col   = choose_first_existing(df0, ["actor_loss", "policy_loss"])
    q_mean_col       = choose_first_existing(df0, ["q_mean", "q_value", "critic_q_mean"])

    curriculum_success_col = choose_first_existing(df0, ["train_curriculum_success"])
    true_success_col       = choose_first_existing(df0, ["train_true_success"])
    legacy_success_col     = choose_first_existing(df0, ["train_success", "train_success_rate"])

    if exists_in_all(return_col):
        make_comparison_plot(dfs, labels, x_col, return_col,
                             title="Episode Return Comparison", ylabel="Return",
                             save_path=os.path.join(outdir, "compare_episode_return.png"),
                             smooth_window=smooth, sparse=False)

    if exists_in_all(eval_return_col):
        make_comparison_plot(dfs, labels, x_col, eval_return_col,
                             title="Evaluation Return Comparison", ylabel="Return",
                             save_path=os.path.join(outdir, "compare_eval_return.png"),
                             smooth_window=max(3, min(smooth, 5)), sparse=True)

    if exists_in_all(eval_success_col):
        make_comparison_plot(dfs, labels, x_col, eval_success_col,
                             title="Evaluation Success Comparison", ylabel="Success Rate",
                             save_path=os.path.join(outdir, "compare_eval_success.png"),
                             smooth_window=max(3, min(smooth, 5)), sparse=True)

    # Curriculum vs true goal comparisons (new schema)
    if exists_in_all(curriculum_success_col):
        make_comparison_plot(dfs, labels, x_col, curriculum_success_col,
                             title="Training Success Comparison (Curriculum Goal)", ylabel="Success Rate",
                             save_path=os.path.join(outdir, "compare_train_curriculum_success.png"),
                             smooth_window=smooth, sparse=False)

    if exists_in_all(true_success_col):
        make_comparison_plot(dfs, labels, x_col, true_success_col,
                             title="Training Success Comparison (True Goal)", ylabel="Success Rate",
                             save_path=os.path.join(outdir, "compare_train_true_success.png"),
                             smooth_window=smooth, sparse=False)

    # Legacy fallback
    if not exists_in_all(curriculum_success_col) and not exists_in_all(true_success_col):
        if exists_in_all(legacy_success_col):
            make_comparison_plot(dfs, labels, x_col, legacy_success_col,
                                 title="Training Success Comparison", ylabel="Success Rate",
                                 save_path=os.path.join(outdir, "compare_train_success.png"),
                                 smooth_window=smooth, sparse=False)

    if exists_in_all(critic_loss_col):
        make_comparison_plot(dfs, labels, x_col, critic_loss_col,
                             title="Critic Loss Comparison", ylabel="Loss",
                             save_path=os.path.join(outdir, "compare_critic_loss.png"),
                             smooth_window=smooth, sparse=False)

    if exists_in_all(actor_loss_col):
        make_comparison_plot(dfs, labels, x_col, actor_loss_col,
                             title="Actor Loss Comparison", ylabel="Loss",
                             save_path=os.path.join(outdir, "compare_actor_loss.png"),
                             smooth_window=smooth, sparse=False)

    if exists_in_all(q_mean_col):
        make_comparison_plot(dfs, labels, x_col, q_mean_col,
                             title="Mean Q Value Comparison", ylabel="Q Mean",
                             save_path=os.path.join(outdir, "compare_q_mean.png"),
                             smooth_window=smooth, sparse=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True,
                        help="One or more training log CSV files")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Optional labels for each CSV (same order as --csv)")
    parser.add_argument("--outdir", type=str, default="plots",
                        help="Base directory where plots will be saved. "
                             "Single-run plots go into a subdirectory named after the CSV file. "
                             "Comparison plots (multiple CSVs) go into --outdir/compare/.")
    parser.add_argument("--smooth", type=int, default=10,
                        help="Moving average window")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    labels = args.labels if args.labels is not None else [
        os.path.splitext(os.path.basename(p))[0] for p in args.csv
    ]
    if len(labels) != len(args.csv):
        raise ValueError("Number of --labels must match number of --csv files.")

    dfs = []
    for path in args.csv:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"{path} is empty.")
        df["__index__"] = range(len(df))
        dfs.append(df)

    x_col = find_x_column(dfs[0])

    if len(dfs) == 1:
        # Single run: plots go into plots/<csv_stem>/
        run_outdir = outdir_for_csv(args.outdir, args.csv[0])
        plot_single_run(dfs[0], x_col, run_outdir, args.smooth)
        print(f"Single-run plots saved to: {run_outdir}/")
    else:
        # Multiple runs: each run gets its own subfolder, plus a shared compare/ folder.
        for df, csv_path, label in zip(dfs, args.csv, labels):
            run_outdir = outdir_for_csv(args.outdir, csv_path)
            plot_single_run(df, x_col, run_outdir, args.smooth)
            print(f"  Run '{label}' plots saved to: {run_outdir}/")

        compare_outdir = os.path.join(args.outdir, "compare")
        plot_comparison(dfs, labels, x_col, compare_outdir, args.smooth)
        print(f"Comparison plots saved to: {compare_outdir}/")


if __name__ == "__main__":
    main()
