"""
Manual SAC + HER + GoalGAN-style automatic goal sampling for Gymnasium-Robotics Fetch tasks.

This is a practical GoalGAN-style curriculum:
- During early warmup, train on normal environment goals.
- Store achieved goals seen during training.
- Later, sample training goals from achieved goals whose recent success rate is intermediate.
- This creates an automatic curriculum of goals that are neither too easy nor impossible.

Place this file next to:
  her.py
  train_sac_her.py

Run examples are included in the ChatGPT message.
"""

from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pandas as pd
import torch

from her.her import env_compute_reward, flatten_goal_obs, make_transition_from_obs_dict
from her.train_sac_her import (
    GaussianActor,
    MLPCritic,
    TrainStats,
    flatten_obs,
    goal_distance,
    make_env,
    evaluate,
    set_seed,
)

from plot_all_results import plot_from_csv

gym.register_envs(gymnasium_robotics)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32, device=device),
            "acts": torch.tensor(self.acts[idx], dtype=torch.float32, device=device),
            "rews": torch.tensor(self.rews[idx], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "done": torch.tensor(self.done[idx], dtype=torch.float32, device=device),
        }


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        device: torch.device,
        alpha: float = 0.2,
        auto_alpha: bool = False,
        alpha_lr: float = 3e-4,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit
        self.auto_alpha = auto_alpha
        self.target_entropy = -float(act_dim)

        self.actor = GaussianActor(obs_dim, act_dim, hidden_dim, act_limit).to(device)
        self.critic1 = MLPCritic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic2 = MLPCritic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic1_target = MLPCritic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic2_target = MLPCritic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        if auto_alpha:
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self._fixed_alpha = float(alpha)

    @property
    def alpha(self) -> torch.Tensor:
        if self.auto_alpha:
            return self.log_alpha.exp()
        return torch.tensor(self._fixed_alpha, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, mean_action = self.actor.sample(obs_t)
        out = mean_action if deterministic else action
        return torch.clamp(out, -self.act_limit, self.act_limit).cpu().numpy()[0]

    def update(self, batch) -> TrainStats:
        obs, acts, rews, next_obs, done = (
            batch["obs"],
            batch["acts"],
            batch["rews"],
            batch["next_obs"],
            batch["done"],
        )

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            y = rews + self.gamma * (1.0 - done) * target_q

        q1 = self.critic1(obs, acts)
        q2 = self.critic2(obs, acts)
        critic1_loss = torch.nn.functional.mse_loss(q1, y)
        critic2_loss = torch.nn.functional.mse_loss(q2, y)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        new_actions, log_prob, _ = self.actor.sample(obs)
        q_new = torch.min(self.critic1(obs, new_actions), self.critic2(obs, new_actions))
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return TrainStats(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            q_mean=float(q1.mean().item()),
            alpha=float(self.alpha.detach().item()),
        )

    def soft_update(self, net, target_net):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


@dataclass
class GoalRecord:
    goal: np.ndarray
    history: deque


class GoalGANCurriculum:
    """
    GoalGAN-style automatic goal sampler.

    Instead of training an actual GAN, this keeps a pool of achieved goals and
    samples goals with intermediate success rate. This captures the main idea:
    train on goals that are currently learnable but not already solved.
    """

    def __init__(
        self,
        max_goals: int = 5000,
        history_len: int = 20,
        success_low: float = 0.2,
        success_high: float = 0.8,
        noise_std: float = 0.01,
        random_goal_prob: float = 0.2,
    ):
        self.max_goals = max_goals
        self.history_len = history_len
        self.success_low = success_low
        self.success_high = success_high
        self.noise_std = noise_std
        self.random_goal_prob = random_goal_prob
        self.records: List[GoalRecord] = []

    def __len__(self):
        return len(self.records)

    def add_goal(self, goal: np.ndarray):
        goal = np.asarray(goal, dtype=np.float32).copy()
        if len(self.records) >= self.max_goals:
            self.records.pop(0)
        self.records.append(GoalRecord(goal=goal, history=deque(maxlen=self.history_len)))

    def _score(self, rec: GoalRecord) -> float:
        if len(rec.history) == 0:
            return 0.5
        return float(np.mean(rec.history))

    def sample(self, env) -> Tuple[Optional[np.ndarray], str]:
        if len(self.records) == 0:
            return None, "env"

        if np.random.rand() < self.random_goal_prob:
            rec = random.choice(self.records)
            source = "buffer_random"
        else:
            candidates = [
                rec for rec in self.records
                if self.success_low <= self._score(rec) <= self.success_high
            ]
            if len(candidates) == 0:
                rec = random.choice(self.records)
                source = "buffer_fallback"
            else:
                rec = random.choice(candidates)
                source = "buffer_intermediate"

        goal = rec.goal.copy()
        if self.noise_std > 0:
            goal = goal + np.random.normal(0.0, self.noise_std, size=goal.shape).astype(np.float32)

        goal = clip_goal_to_env(env, goal)
        return goal, source

    def update_nearest(self, goal: np.ndarray, success: float):
        if len(self.records) == 0 or goal is None:
            return
        goal = np.asarray(goal, dtype=np.float32)
        dists = [float(np.linalg.norm(rec.goal - goal)) for rec in self.records]
        idx = int(np.argmin(dists))
        self.records[idx].history.append(float(success))

    def stats(self) -> Dict[str, float]:
        if len(self.records) == 0:
            return {
                "goal_buffer_size": 0,
                "goal_success_mean": np.nan,
                "goal_intermediate_frac": np.nan,
            }

        scores = np.array([self._score(rec) for rec in self.records], dtype=np.float32)
        intermediate = np.logical_and(scores >= self.success_low, scores <= self.success_high)
        return {
            "goal_buffer_size": len(self.records),
            "goal_success_mean": float(np.mean(scores)),
            "goal_intermediate_frac": float(np.mean(intermediate)),
        }


def clip_goal_to_env(env, goal: np.ndarray) -> np.ndarray:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "goal_space"):
        space = unwrapped.goal_space
        if hasattr(space, "low") and hasattr(space, "high"):
            return np.clip(goal, space.low, space.high).astype(np.float32)
    return goal.astype(np.float32)


def set_env_goal(env, goal: np.ndarray, obs_dict: Optional[dict] = None):
    """
    For Fetch envs, changing obs_dict alone is not enough.
    The internal env goal must also be changed.
    """
    goal = np.asarray(goal, dtype=np.float32).copy()
    unwrapped = env.unwrapped

    if hasattr(unwrapped, "goal"):
        unwrapped.goal = goal.copy()
    elif hasattr(unwrapped, "_goal"):
        unwrapped._goal = goal.copy()
    else:
        raise AttributeError("Could not find env.unwrapped.goal. This script expects a Fetch-style goal env.")

    if obs_dict is not None:
        obs_dict["desired_goal"] = goal.copy()
    return obs_dict


def add_episode_with_her(replay_buffer, episode, reward_fn, her_k: int, use_her: bool, future_offset: int) -> int:
    her_added = 0
    T = len(episode)

    for t, tr in enumerate(episode):
        obs = flatten_goal_obs(tr.observation, tr.achieved_goal, tr.desired_goal)
        next_obs = flatten_goal_obs(tr.next_observation, tr.next_achieved_goal, tr.desired_goal)
        replay_buffer.add(obs, tr.action, tr.reward, next_obs, tr.done)

        if not use_her or her_k <= 0:
            continue

        low = min(T - 1, t + future_offset)
        if low >= T:
            continue

        for _ in range(her_k):
            future_t = np.random.randint(low, T)
            new_goal = episode[future_t].next_achieved_goal.copy()
            relabeled_reward = reward_fn(tr.next_achieved_goal, new_goal, tr.info)

            relabeled_obs = flatten_goal_obs(tr.observation, tr.achieved_goal, new_goal)
            relabeled_next_obs = flatten_goal_obs(tr.next_observation, tr.next_achieved_goal, new_goal)
            replay_buffer.add(relabeled_obs, tr.action, relabeled_reward, relabeled_next_obs, tr.done)
            her_added += 1

    return her_added


def create_run_dirs(base_dir: str, algorithm: str, task: str, reward_type: str, seed: int) -> dict:
    run_group = f"{task}_{reward_type}"
    seed_name = f"seed{seed}"
    dirs = {
        "logs": os.path.join(base_dir, "logs", algorithm, run_group, seed_name),
        "plots": os.path.join(base_dir, "plots", algorithm, run_group, seed_name),
        "models": os.path.join(base_dir, "models", algorithm, run_group, seed_name),
        "summaries": os.path.join(base_dir, "summaries", algorithm, run_group, seed_name),
        "results": os.path.join(base_dir, "results"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def simple_plot_from_csv(csv_path: str, outdir: str, smooth: int = 10):
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)

    def moving_avg(x, w):
        s = pd.Series(x)
        return s.rolling(w, min_periods=1).mean().to_numpy()

    plots = [
        ("train_true_success", "Train true success", "train_true_success.png"),
        ("train_true_goal_distance", "Train true goal distance", "train_true_goal_distance.png"),
        ("eval_success", "Eval success", "eval_success.png"),
        ("eval_true_goal_distance", "Eval true goal distance", "eval_true_goal_distance.png"),
        ("goal_intermediate_frac", "Fraction of intermediate goals", "goal_intermediate_frac.png"),
    ]

    for col, title, fname in plots:
        if col not in df.columns:
            continue
        y = df[col].to_numpy(dtype=float)
        x = df["episode"].to_numpy()
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            continue

        plt.figure()
        plt.plot(x[mask], y[mask], label="raw")
        if smooth > 1:
            plt.plot(x[mask], moving_avg(y[mask], smooth), label=f"MA({smooth})")
        plt.xlabel("Episode")
        plt.ylabel(col)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close()


def summarize_run(csv_path: str, algorithm: str, args, model_path: str, last_n: int = 5) -> dict:
    df = pd.read_csv(csv_path)
    eval_success = df["eval_success"].dropna() if "eval_success" in df.columns else pd.Series(dtype=float)
    eval_return = df["eval_return"].dropna() if "eval_return" in df.columns else pd.Series(dtype=float)
    eval_dist = df["eval_true_goal_distance"].dropna() if "eval_true_goal_distance" in df.columns else pd.Series(dtype=float)

    return {
        "algorithm": algorithm,
        "task": args.task,
        "reward_type": args.reward_type,
        "seed": args.seed,
        "episodes": args.episodes,
        "use_her": bool(args.use_her),
        "her_k": args.her_k,
        "use_goalgan": bool(args.use_goalgan),
        "goalgan_warmup_episodes": args.goalgan_warmup_episodes,
        "goalgan_success_low": args.goalgan_success_low,
        "goalgan_success_high": args.goalgan_success_high,
        "final_eval_success": float(eval_success.iloc[-1]) if len(eval_success) else float("nan"),
        "best_eval_success": float(eval_success.max()) if len(eval_success) else float("nan"),
        "last5_eval_success_mean": float(eval_success.tail(last_n).mean()) if len(eval_success) else float("nan"),
        "final_eval_return": float(eval_return.iloc[-1]) if len(eval_return) else float("nan"),
        "last5_eval_distance_mean": float(eval_dist.tail(last_n).mean()) if len(eval_dist) else float("nan"),
        "csv_path": csv_path,
        "model_path": model_path,
    }


def write_summary_files(summary: dict, run_summary_dir: str, global_results_dir: str) -> None:
    pd.DataFrame([summary]).to_csv(os.path.join(run_summary_dir, "summary.csv"), index=False)
    with open(os.path.join(run_summary_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    global_csv = os.path.join(global_results_dir, "goalgan_her_results.csv")
    new_row = pd.DataFrame([summary])
    if os.path.exists(global_csv):
        old = pd.read_csv(global_csv)
        key = (
            (old["algorithm"] == summary["algorithm"])
            & (old["task"] == summary["task"])
            & (old["reward_type"] == summary["reward_type"])
            & (old["seed"] == summary["seed"])
        )
        old = old.loc[~key]
        out = pd.concat([old, new_row], ignore_index=True)
    else:
        out = new_row

    out.sort_values(["task", "reward_type", "algorithm", "seed"]).to_csv(global_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Manual SAC + HER + GoalGAN-style automatic goal sampling for Fetch tasks.")

    parser.add_argument("--task", type=str, default="FetchReach", choices=["FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace"])
    parser.add_argument("--reward-type", type=str, default="sparse", choices=["dense", "sparse"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)

    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--updates-per-step", type=int, default=1)

    parser.add_argument("--use-her", action="store_true")
    parser.add_argument("--her-k", type=int, default=4)
    parser.add_argument("--her-future-offset", type=int, default=1)

    parser.add_argument("--use-goalgan", action="store_true")
    parser.add_argument("--goalgan-warmup-episodes", type=int, default=100)
    parser.add_argument("--goalgan-max-goals", type=int, default=5000)
    parser.add_argument("--goalgan-history-len", type=int, default=20)
    parser.add_argument("--goalgan-success-low", type=float, default=0.2)
    parser.add_argument("--goalgan-success-high", type=float, default=0.8)
    parser.add_argument("--goalgan-noise-std", type=float, default=0.01)
    parser.add_argument("--goalgan-random-goal-prob", type=float, default=0.2)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-on-first-episode", action="store_true")
    parser.add_argument("--carry-eval-forward", action="store_true")
    parser.add_argument("--plot-smooth", type=int, default=10)
    parser.add_argument("--goalgan-dir", type=str, default="goalgan_her")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    if args.use_goalgan and args.use_her:
        algorithm = "sac_her_goalgan"
    elif args.use_goalgan:
        algorithm = "sac_goalgan"
    elif args.use_her:
        algorithm = "sac_her"
    else:
        algorithm = "sac"

    env = make_env(args.task, args.reward_type, args.seed)
    obs_dict, _ = env.reset(seed=args.seed)

    obs = flatten_obs(obs_dict)
    obs_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        alpha_lr=args.alpha_lr,
    )

    replay = ReplayBuffer(obs_dim, act_dim, args.replay_size)

    curriculum = GoalGANCurriculum(
        max_goals=args.goalgan_max_goals,
        history_len=args.goalgan_history_len,
        success_low=args.goalgan_success_low,
        success_high=args.goalgan_success_high,
        noise_std=args.goalgan_noise_std,
        random_goal_prob=args.goalgan_random_goal_prob,
    )

    run_dirs = create_run_dirs(args.goalgan_dir, algorithm, args.task, args.reward_type, args.seed)
    run_name = f"{algorithm}_{args.task}_{args.reward_type}_seed{args.seed}"
    csv_path = os.path.join(run_dirs["logs"], f"{run_name}.csv")
    model_path = os.path.join(run_dirs["models"], f"{run_name}_final.pt")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "total_steps",
            "train_return",
            "train_true_success",
            "train_true_goal_distance",
            "eval_return",
            "eval_success",
            "eval_true_goal_distance",
            "critic_loss",
            "actor_loss",
            "q_mean",
            "alpha",
            "her_transitions_added",
            "replay_size",
            "goal_source",
            "goal_buffer_size",
            "goal_success_mean",
            "goal_intermediate_frac",
        ])

        total_steps = 0
        recent_success = deque(maxlen=10)
        last_eval_return = np.nan
        last_eval_success = np.nan
        last_eval_distance = np.nan

        for episode_idx in range(1, args.episodes + 1):
            obs_dict, _ = env.reset()

            sampled_goal = None
            goal_source = "env"

            if args.use_goalgan and episode_idx > args.goalgan_warmup_episodes and len(curriculum) > 0:
                sampled_goal, goal_source = curriculum.sample(env)
                if sampled_goal is not None:
                    obs_dict = set_env_goal(env, sampled_goal, obs_dict)

            true_goal = obs_dict["desired_goal"].copy()

            done = False
            truncated = False
            ep_return = 0.0
            ep_success = 0.0
            ep_distance = np.nan
            episode_transitions = []
            episode_steps = 0

            critic_losses = []
            actor_losses = []
            q_means = []
            alphas = []

            while not (done or truncated):
                obs = flatten_obs(obs_dict)

                if total_steps < args.start_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs, deterministic=False)

                next_obs_dict, reward, done, truncated, info = env.step(action)

                # Make sure returned desired_goal matches the sampled curriculum goal.
                if sampled_goal is not None:
                    next_obs_dict["desired_goal"] = true_goal.copy()

                terminal = float(done)

                episode_transitions.append(
                    make_transition_from_obs_dict(
                        obs_dict,
                        action,
                        reward,
                        next_obs_dict,
                        terminal,
                        info,
                    )
                )

                ep_return += float(reward)
                ep_success = max(ep_success, float(info.get("is_success", 0.0)))
                ep_distance = goal_distance(next_obs_dict["achieved_goal"], true_goal)

                # Store achieved goals for future curriculum sampling.
                if args.use_goalgan:
                    curriculum.add_goal(next_obs_dict["achieved_goal"])

                obs_dict = next_obs_dict
                total_steps += 1
                episode_steps += 1

            if args.use_goalgan and sampled_goal is not None:
                curriculum.update_nearest(sampled_goal, ep_success)

            her_added = add_episode_with_her(
                replay,
                episode_transitions,
                lambda ag, dg, info: env_compute_reward(env, ag, dg, info),
                args.her_k,
                args.use_her,
                args.her_future_offset,
            )

            if replay.size >= args.batch_size:
                for _ in range(max(1, episode_steps * args.updates_per_step)):
                    batch = replay.sample(args.batch_size, device)
                    stats = agent.update(batch)
                    critic_losses.append(stats.critic_loss)
                    actor_losses.append(stats.actor_loss)
                    q_means.append(stats.q_mean)
                    alphas.append(stats.alpha)

            recent_success.append(ep_success)

            should_eval = (episode_idx % args.eval_every == 0) or (episode_idx == 1 and args.eval_on_first_episode)
            if should_eval:
                # Evaluation stays on the original environment goal distribution.
                last_eval_return, last_eval_success, last_eval_distance = evaluate(
                    agent,
                    args.task,
                    args.reward_type,
                    args.seed,
                    args.eval_episodes,
                )

            eval_return = last_eval_return if (should_eval or args.carry_eval_forward) else np.nan
            eval_success = last_eval_success if (should_eval or args.carry_eval_forward) else np.nan
            eval_distance = last_eval_distance if (should_eval or args.carry_eval_forward) else np.nan

            goal_stats = curriculum.stats()

            writer.writerow([
                episode_idx,
                total_steps,
                ep_return,
                ep_success,
                ep_distance,
                eval_return,
                eval_success,
                eval_distance,
                float(np.mean(critic_losses)) if critic_losses else np.nan,
                float(np.mean(actor_losses)) if actor_losses else np.nan,
                float(np.mean(q_means)) if q_means else np.nan,
                float(np.mean(alphas)) if alphas else float(agent.alpha.detach().item()),
                her_added,
                replay.size,
                goal_source,
                goal_stats["goal_buffer_size"],
                goal_stats["goal_success_mean"],
                goal_stats["goal_intermediate_frac"],
            ])
            f.flush()

            print(
                f"Episode {episode_idx:04d} | algo={algorithm} | task={args.task} | reward={args.reward_type} | "
                f"return={ep_return:.2f} | train_success={ep_success:.1f} "
                f"(recent={float(np.mean(recent_success)):.3f}) | dist={ep_distance:.4f} | "
                f"eval_success={(eval_success if not np.isnan(eval_success) else float('nan')):.3f} | "
                f"goal={goal_source} | goal_buf={goal_stats['goal_buffer_size']} | "
                f"intermediate={goal_stats['goal_intermediate_frac']:.3f} | HER+={her_added} | replay={replay.size}"
            )

    env.close()

    torch.save(
        {
            "algorithm": algorithm,
            "args": vars(args),
            "actor_state_dict": agent.actor.state_dict(),
        },
        model_path,
    )

    summary = summarize_run(csv_path, algorithm, args, model_path)
    write_summary_files(summary, run_dirs["summaries"], run_dirs["results"])

    print(f"Saved log to: {csv_path}")
    print(f"Saved model to: {model_path}")

    plot_from_csv(csv_path, outdir=run_dirs["plots"], smooth=args.plot_smooth)


if __name__ == "__main__":
    main()
