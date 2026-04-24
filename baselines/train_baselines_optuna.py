
import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple

import optuna
import pandas as pd


SCRIPT_MAP = {
    "ddpg": "baseline_ddpg.py",
    "td3": "baseline_td3.py",
    "sac": "baseline_sac.py",
}


def build_common_cmd(python_exec: str, script_name: str, args, seed: int, log_dir: str, plots_dir: str) -> List[str]:
    cmd = [
        python_exec,
        script_name,
        "--task", args.task,
        "--reward-type", args.reward_type,
        "--episodes", str(args.episodes),
        "--seed", str(seed),
        "--device", args.device,
        "--hidden-dim", str(args.hidden_dim),
        "--replay-size", str(args.replay_size),
        "--batch-size", str(args.batch_size),
        "--actor-lr", str(args.actor_lr),
        "--critic-lr", str(args.critic_lr),
        "--gamma", str(args.gamma),
        "--tau", str(args.tau),
        "--start-steps", str(args.start_steps),
        "--updates-per-step", str(args.updates_per_step),
        "--eval-every", str(args.eval_every),
        "--eval-episodes", str(args.eval_episodes),
        "--log-dir", log_dir,
        "--plots-dir", plots_dir,
        "--plot-smooth", str(args.plot_smooth),
        "--baseline-dir", args.baseline_dir,
    ]

    if args.eval_on_first_episode:
        cmd.append("--eval-on-first-episode")
    if args.carry_eval_forward:
        cmd.append("--carry-eval-forward")

    return cmd


def add_algorithm_specific_args(cmd: List[str], algorithm: str, args) -> List[str]:
    cmd = list(cmd)
    if algorithm in {"ddpg", "td3"}:
        cmd.extend(["--action-noise", str(args.action_noise)])
    if algorithm == "td3":
        cmd.extend([
            "--td3-policy-noise", str(args.td3_policy_noise),
            "--td3-noise-clip", str(args.td3_noise_clip),
            "--td3-policy-delay", str(args.td3_policy_delay),
        ])
    if algorithm == "sac":
        cmd.extend(["--alpha", str(args.alpha)])
    return cmd


def csv_name_for_run(algorithm: str, task: str, reward_type: str, seed: int) -> str:
    return f"{algorithm}_{task}_{reward_type}_seed{seed}.csv"


def csv_path_for_run(baseline_dir: str, algorithm: str, task: str, reward_type: str, seed: int) -> str:
    return os.path.join(
        baseline_dir,
        "logs",
        algorithm,
        f"{task}_{reward_type}",
        f"seed{seed}",
        csv_name_for_run(algorithm, task, reward_type, seed),
    )


def read_objective_from_csv(csv_path: str, objective: str) -> float:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    if objective not in df.columns:
        raise ValueError(f"Objective column '{objective}' not found in {csv_path}. Columns: {list(df.columns)}")

    valid = df[objective].dropna()
    if len(valid) == 0:
        raise ValueError(f"No valid values found for objective '{objective}' in {csv_path}")

    return float(valid.max())


def suggest_params(trial: optuna.Trial, algorithm: str) -> Dict:
    params = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
        "critic_lr": trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 1e-3, 2e-2, log=True),
        "start_steps": trial.suggest_categorical("start_steps", [1000, 5000, 10000]),
        "updates_per_step": trial.suggest_categorical("updates_per_step", [1, 2]),
        "replay_size": trial.suggest_categorical("replay_size", [200000, 500000, 1000000]),
    }

    if algorithm in {"ddpg", "td3"}:
        params["action_noise"] = trial.suggest_float("action_noise", 0.05, 0.3)

    if algorithm == "td3":
        params["td3_policy_noise"] = trial.suggest_float("td3_policy_noise", 0.1, 0.4)
        params["td3_noise_clip"] = trial.suggest_float("td3_noise_clip", 0.2, 0.7)
        params["td3_policy_delay"] = trial.suggest_categorical("td3_policy_delay", [2, 3])

    if algorithm == "sac":
        params["alpha"] = trial.suggest_float("alpha", 0.05, 0.4)

    return params


def apply_trial_params(args, params: Dict) -> None:
    for key, value in params.items():
        setattr(args, key, value)


def run_single_trial(args, trial: optuna.Trial) -> float:
    algorithm = args.algorithm
    params = suggest_params(trial, algorithm)
    apply_trial_params(args, params)

    python_exec = sys.executable
    script_name = SCRIPT_MAP[algorithm]

    seed_scores = []
    for seed in args.seeds:
        trial_log_dir = os.path.join(args.log_dir, f"trial_{trial.number}")
        trial_plots_dir = os.path.join(args.plots_dir, f"trial_{trial.number}")
        os.makedirs(trial_log_dir, exist_ok=True)
        os.makedirs(trial_plots_dir, exist_ok=True)

        cmd = build_common_cmd(
            python_exec=python_exec,
            script_name=script_name,
            args=args,
            seed=seed,
            log_dir=trial_log_dir,
            plots_dir=trial_plots_dir,
        )
        cmd = add_algorithm_specific_args(cmd, algorithm, args)

        print("\n" + "=" * 100)
        print(f"TRIAL {trial.number} | seed={seed} | algorithm={algorithm}")
        print(" ".join(cmd))
        print("=" * 100)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Baseline script failed on trial {trial.number}, seed {seed}")

        csv_path = csv_path_for_run(args.baseline_dir, algorithm, args.task, args.reward_type, seed)
        score = read_objective_from_csv(csv_path, args.objective)
        seed_scores.append(score)

        trial.report(float(sum(seed_scores) / len(seed_scores)), step=len(seed_scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_score = float(sum(seed_scores) / len(seed_scores))
    return mean_score


def save_study_summary(study: optuna.Study, out_path: str) -> None:
    summary = {
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_trial.params,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Optuna tuner for DDPG / TD3 / SAC baseline scripts.")
    parser.add_argument("--algorithm", choices=["ddpg", "td3", "sac"], required=True)
    parser.add_argument("--task", choices=["FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace"], default="FetchReach")
    parser.add_argument("--reward-type", dest="reward_type", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-on-first-episode", action="store_true")
    parser.add_argument("--carry-eval-forward", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--updates-per-step", type=int, default=1)

    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--td3-policy-noise", type=float, default=0.2)
    parser.add_argument("--td3-noise-clip", type=float, default=0.5)
    parser.add_argument("--td3-policy-delay", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.2)

    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", type=str, default="baseline_optuna")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optional Optuna storage URL, e.g. sqlite:///optuna.db")
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--objective", type=str, default="eval_success",
                        help="CSV column to maximize, usually eval_success or eval_return")
    parser.add_argument("--log-dir", type=str, default="logs_optuna")
    parser.add_argument("--plots-dir", type=str, default="plots_optuna")
    parser.add_argument("--plot-smooth", type=int, default=10)
    parser.add_argument("--results-dir", type=str, default="baselines/optuna_results")
    parser.add_argument("--baseline-dir", type=str, default="baselines")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(lambda trial: run_single_trial(args, trial), n_trials=args.n_trials)

    print("\nBest trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  value:  {study.best_value}")
    print("  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    csv_trials_path = os.path.join(args.results_dir, f"{args.study_name}_trials.csv")
    json_summary_path = os.path.join(args.results_dir, f"{args.study_name}_best.json")

    study.trials_dataframe().to_csv(csv_trials_path, index=False)
    save_study_summary(study, json_summary_path)

    print(f"\nSaved trials table to: {csv_trials_path}")
    print(f"Saved best summary to: {json_summary_path}")


if __name__ == "__main__":
    main()
