"""
Run Optuna hyperparameter search for SAC + HER + HIRO on all Fetch tasks
and on both dense and sparse rewards.

This script is a launcher around optuna_sac_hiro_her.py.
It runs one Optuna study for each (task, reward_type) pair and stores logs,
plots, Optuna DBs, and summaries in separate folders.

Example:
  python run_optuna_all_rewards_sac_hiro_her.py \
    --tasks FetchReach FetchPush FetchSlide FetchPickAndPlace \
    --reward-types dense sparse \
    --n-trials 20 \
    --seeds 0 1 \
    --episodes 300 \
    --device cpu
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_TASKS = [
    "FetchReach",
    "FetchPush",
    "FetchSlide",
    "FetchPickAndPlace",
]

DEFAULT_REWARD_TYPES = ["dense", "sparse"]


def slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def build_command(args: argparse.Namespace, task: str, reward_type: str) -> List[str]:
    task_slug = slug(task)
    reward_slug = slug(reward_type)

    run_name = f"{task_slug}_{reward_slug}"
    log_dir = Path(args.log_dir) / run_name
    plots_dir = Path(args.plots_dir) / run_name
    results_dir = Path(args.results_dir) / run_name

    log_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"sac_her_hiro_{task}_{reward_type}"

    cmd = [
        sys.executable,
        args.optuna_script,
        "--task", task,
        "--reward-type", reward_type,
        "--use-her",
        "--use-hiro",
        "--hiro-high-level-mode", args.hiro_high_level_mode,
        "--n-trials", str(args.n_trials),
        "--episodes", str(args.episodes),
        "--device", args.device,
        "--eval-every", str(args.eval_every),
        "--eval-episodes", str(args.eval_episodes),
        "--objective", args.objective,
        "--final-last-k", str(args.final_last_k),
        "--study-name", study_name,
        "--sampler-seed", str(args.sampler_seed),
        "--log-dir", str(log_dir),
        "--plots-dir", str(plots_dir),
        "--results-dir", str(results_dir),
        "--plot-smooth", str(args.plot_smooth),
    ]

    if args.seeds:
        cmd.append("--seeds")
        cmd.extend(str(seed) for seed in args.seeds)

    if args.eval_on_first_episode:
        cmd.append("--eval-on-first-episode")
    if args.carry_eval_forward:
        cmd.append("--carry-eval-forward")

    if args.storage_dir:
        storage_dir = Path(args.storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / f"{study_name}.db"
        cmd.extend(["--storage", f"sqlite:///{storage_path}"])

    return cmd


def run_one_search(args: argparse.Namespace, task: str, reward_type: str) -> dict:
    cmd = build_command(args, task, reward_type)

    print("\n" + "=" * 100)
    print(
        f"STARTING OPTUNA SEARCH | task={task} | reward={reward_type} | "
        f"method=SAC+HER+HIRO"
    )
    print(" ".join(cmd))
    print("=" * 100)

    result = subprocess.run(cmd, check=False)
    status = "success" if result.returncode == 0 else "failed"

    print("\n" + "=" * 100)
    print(
        f"FINISHED | task={task} | reward={reward_type} | "
        f"status={status} | returncode={result.returncode}"
    )
    print("=" * 100)

    if result.returncode != 0 and not args.continue_on_error:
        raise RuntimeError(
            f"Optuna search failed for task={task}, reward={reward_type}. "
            "Use --continue-on-error to continue after failures."
        )

    return {
        "task": task,
        "reward_type": reward_type,
        "method": "SAC+HER+HIRO",
        "status": status,
        "returncode": result.returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch Optuna SAC+HER+HIRO searches on all selected Fetch tasks "
            "and reward types."
        )
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        choices=DEFAULT_TASKS,
        help="Fetch tasks to run. Default: all standard Fetch tasks.",
    )
    parser.add_argument(
        "--reward-types",
        nargs="+",
        default=DEFAULT_REWARD_TYPES,
        choices=DEFAULT_REWARD_TYPES,
        help="Reward types to run. Default: dense sparse.",
    )
    parser.add_argument(
        "--optuna-script",
        type=str,
        default="optuna_sac_hiro_her.py",
        help="Path to optuna_sac_hiro_her.py.",
    )

    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-on-first-episode", action="store_true")
    parser.add_argument("--carry-eval-forward", action="store_true")

    parser.add_argument("--objective", type=str, default="eval_success")
    parser.add_argument(
        "--final-last-k",
        type=int,
        default=0,
        help=(
            "If >0, optimize the mean of the last K non-NaN objective values "
            "instead of the max objective value."
        ),
    )

    parser.add_argument(
        "--hiro-high-level-mode",
        choices=["heuristic", "learned"],
        default="heuristic",
        help="Use heuristic first for reliability; learned is more expensive.",
    )
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--plot-smooth", type=int, default=10)

    parser.add_argument("--log-dir", type=str, default="logs_optuna_all_rewards")
    parser.add_argument("--plots-dir", type=str, default="plots_optuna_all_rewards")
    parser.add_argument("--results-dir", type=str, default="optuna_results_all_rewards")
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="optuna_storage_all_rewards",
        help="Directory for sqlite Optuna DBs. Set empty string to disable storage.",
    )
    parser.add_argument("--continue-on-error", action="store_true")

    args = parser.parse_args()

    if args.storage_dir == "":
        args.storage_dir = None

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    summaries = []
    for task in args.tasks:
        for reward_type in args.reward_types:
            summaries.append(run_one_search(args, task, reward_type))

    summary_path = Path(args.results_dir) / "all_rewards_sac_her_hiro_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\nAll requested searches finished.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
