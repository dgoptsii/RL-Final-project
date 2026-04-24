from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

import optuna
import pandas as pd

TRAIN_SCRIPT = "train_sac_hiro_her.py"


def csv_name(args, seed: int) -> str:
    method = "sac"
    if args.use_her:
        method += "_her"
    if args.use_hiro:
        method += "_hiro"
    return f"{method}_{args.task}_{args.reward_type}_seed{seed}.csv"


def suggest_params(trial: optuna.Trial, args) -> Dict:
    params = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
        "critic_lr": trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 1e-3, 2e-2, log=True),
        "alpha": trial.suggest_float("alpha", 0.05, 0.4),
        "start_steps": trial.suggest_categorical("start_steps", [1000, 5000, 10000]),
        "updates_per_step": trial.suggest_categorical("updates_per_step", [1, 2]),
        "replay_size": trial.suggest_categorical("replay_size", [200000, 500000, 1000000]),
    }
    if args.use_her:
        params["her_k"] = trial.suggest_categorical("her_k", [2, 4, 8])
        params["her_future_offset"] = trial.suggest_categorical("her_future_offset", [0, 1])
    if args.use_hiro:
        params["hiro_interval"] = trial.suggest_categorical("hiro_interval", [5, 10, 20])
        params["hiro_subgoal_ratio"] = trial.suggest_float("hiro_subgoal_ratio", 0.2, 0.9)
        params["hiro_subgoal_noise"] = trial.suggest_float("hiro_subgoal_noise", 0.0, 0.05)
        params["hiro_subgoal_limit"] = trial.suggest_float("hiro_subgoal_limit", 0.1, 0.5)
        if args.hiro_high_level_mode == "learned":
            params["hiro_high_hidden_dim"] = trial.suggest_categorical("hiro_high_hidden_dim", [128, 256])
            params["hiro_high_actor_lr"] = trial.suggest_float("hiro_high_actor_lr", 1e-5, 1e-3, log=True)
            params["hiro_high_critic_lr"] = trial.suggest_float("hiro_high_critic_lr", 1e-5, 1e-3, log=True)
            params["hiro_high_noise"] = trial.suggest_float("hiro_high_noise", 0.01, 0.2)
            params["hiro_high_start_steps"] = trial.suggest_categorical("hiro_high_start_steps", [500, 1000, 3000])
    return params


def build_cmd(args, params: Dict, seed: int, trial_log_dir: str, trial_plots_dir: str, trial_checkpoint_dir: str) -> List[str]:
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--task", args.task,
        "--reward-type", args.reward_type,
        "--episodes", str(args.episodes),
        "--seed", str(seed),
        "--device", args.device,
        "--eval-every", str(args.eval_every),
        "--eval-episodes", str(args.eval_episodes),
    ]
    if args.use_her:
        cmd.append("--use-her")
    if args.use_hiro:
        cmd.append("--use-hiro")
        cmd.extend(["--hiro-high-level-mode", args.hiro_high_level_mode])

    fixed = {
        "log_dir": trial_log_dir,
        "plots_dir": trial_plots_dir,
        "checkpoint_dir": trial_checkpoint_dir,
        "plot_smooth": args.plot_smooth,
    }
    all_params = {**params, **fixed}
    key_to_arg = {
        "hidden_dim": "--hidden-dim",
        "replay_size": "--replay-size",
        "batch_size": "--batch-size",
        "actor_lr": "--actor-lr",
        "critic_lr": "--critic-lr",
        "gamma": "--gamma",
        "tau": "--tau",
        "alpha": "--alpha",
        "start_steps": "--start-steps",
        "updates_per_step": "--updates-per-step",
        "her_k": "--her-k",
        "her_future_offset": "--her-future-offset",
        "hiro_interval": "--hiro-interval",
        "hiro_subgoal_ratio": "--hiro-subgoal-ratio",
        "hiro_subgoal_noise": "--hiro-subgoal-noise",
        "hiro_subgoal_limit": "--hiro-subgoal-limit",
        "hiro_high_hidden_dim": "--hiro-high-hidden-dim",
        "hiro_high_actor_lr": "--hiro-high-actor-lr",
        "hiro_high_critic_lr": "--hiro-high-critic-lr",
        "hiro_high_noise": "--hiro-high-noise",
        "hiro_high_start_steps": "--hiro-high-start-steps",
        "log_dir": "--log-dir",
        "plots_dir": "--plots-dir",
        "checkpoint_dir": "--checkpoint-dir",
        "plot_smooth": "--plot-smooth",
    }
    for key, value in all_params.items():
        if key in key_to_arg:
            cmd.extend([key_to_arg[key], str(value)])
    if args.eval_on_first_episode:
        cmd.append("--eval-on-first-episode")
    if args.carry_eval_forward:
        cmd.append("--carry-eval-forward")
    return cmd


def read_objective(csv_path: str, objective: str, final_last_k: int) -> float:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if objective not in df.columns:
        raise ValueError(f"Column {objective} not in {csv_path}. Columns: {list(df.columns)}")
    values = df[objective].dropna()
    if len(values) == 0:
        raise ValueError(f"No non-NaN objective values for {objective} in {csv_path}")
    if final_last_k > 0:
        return float(values.tail(final_last_k).mean())
    return float(values.max())


def run_trial(args, trial: optuna.Trial) -> float:
    params = suggest_params(trial, args)
    seed_scores = []
    for seed in args.seeds:
        trial_log_dir = os.path.join(args.log_dir, f"trial_{trial.number}")
        trial_plots_dir = os.path.join(args.plots_dir, f"trial_{trial.number}")
        trial_checkpoint_dir = os.path.join(args.checkpoint_dir, f"trial_{trial.number}")
        os.makedirs(trial_log_dir, exist_ok=True)
        os.makedirs(trial_plots_dir, exist_ok=True)
        os.makedirs(trial_checkpoint_dir, exist_ok=True)

        cmd = build_cmd(args, params, seed, trial_log_dir, trial_plots_dir, trial_checkpoint_dir)
        print("\n" + "=" * 100)
        print(f"TRIAL {trial.number} | seed={seed} | HER={args.use_her} | HIRO={args.use_hiro}")
        print(" ".join(cmd))
        print("=" * 100)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed for trial={trial.number}, seed={seed}")
        score = read_objective(os.path.join(trial_log_dir, csv_name(args, seed)), args.objective, args.final_last_k)
        seed_scores.append(score)
        trial.report(float(sum(seed_scores) / len(seed_scores)), step=len(seed_scores))
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(sum(seed_scores) / len(seed_scores))


def save_summary(study: optuna.Study, out_path: str, args) -> None:
    summary = {
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_trial.params,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "use_her": args.use_her,
        "use_hiro": args.use_hiro,
        "task": args.task,
        "reward_type": args.reward_type,
        "objective": args.objective,
        "final_last_k": args.final_last_k,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Optuna search for SAC / SAC+HER / SAC+HIRO / SAC+HIRO+HER.")
    parser.add_argument("--task", choices=["FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace"], default="FetchReach")
    parser.add_argument("--reward-type", dest="reward_type", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--use-her", action="store_true")
    parser.add_argument("--use-hiro", action="store_true")
    parser.add_argument("--hiro-high-level-mode", choices=["heuristic", "learned"], default="heuristic")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-on-first-episode", action="store_true")
    parser.add_argument("--carry-eval-forward", action="store_true")
    parser.add_argument("--objective", type=str, default="eval_success")
    parser.add_argument("--final-last-k", type=int, default=0)
    parser.add_argument("--study-name", type=str, default="sac_hiro_her_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="logs_optuna_hiro_her")
    parser.add_argument("--plots-dir", type=str, default="plots_optuna_hiro_her")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_optuna_hiro_her")
    parser.add_argument("--results-dir", type=str, default="optuna_results_hiro_her")
    parser.add_argument("--plot-smooth", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
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
    study.optimize(lambda trial: run_trial(args, trial), n_trials=args.n_trials)

    print("\nBest trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  value:  {study.best_value}")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    suffix = f"{args.task}_{args.reward_type}"
    if args.use_her:
        suffix += "_her"
    if args.use_hiro:
        suffix += "_hiro"
    trials_path = os.path.join(args.results_dir, f"{args.study_name}_{suffix}_trials.csv")
    summary_path = os.path.join(args.results_dir, f"{args.study_name}_{suffix}_best.json")
    study.trials_dataframe().to_csv(trials_path, index=False)
    save_summary(study, summary_path, args)
    print(f"Saved trials table to: {trials_path}")
    print(f"Saved best summary to: {summary_path}")


if __name__ == "__main__":
    main()
