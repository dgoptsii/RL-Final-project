"""
Manual SAC + prioritized HER for Gymnasium-Robotics Fetch tasks.

This keeps your existing HER relabeling idea, but samples replay transitions with
priorities based on TD error. High-error transitions are sampled more often, and
importance-sampling weights reduce bias in the critic loss.

Place this file next to:
  her.py, train_sac_her.py, plot_prioritized_her_results.py
"""
from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from her import env_compute_reward, flatten_goal_obs, make_transition_from_obs_dict
from plot_all_results import plot_from_csv
from train_sac_her import (
    GaussianActor,
    MLPCritic,
    TrainStats,
    flatten_obs,
    goal_distance,
    make_env,
    evaluate,
    set_seed,
)

gym.register_envs(gymnasium_robotics)


class PrioritizedReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, alpha: float = 0.6, eps: float = 1e-6):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size
        self.alpha = alpha
        self.eps = eps
        self.max_priority = 1.0

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.priorities[self.ptr] = self.max_priority
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device, beta: float = 0.4):
        if self.size < batch_size:
            raise ValueError(f"Replay buffer has {self.size} transitions, needs {batch_size}.")
        priorities = self.priorities[:self.size]
        scaled = np.power(priorities + self.eps, self.alpha)
        probs = scaled / scaled.sum()
        idx = np.random.choice(self.size, size=batch_size, replace=True, p=probs)
        weights = np.power(self.size * probs[idx], -beta)
        weights = weights / weights.max()
        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32, device=device),
            "acts": torch.tensor(self.acts[idx], dtype=torch.float32, device=device),
            "rews": torch.tensor(self.rews[idx], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "done": torch.tensor(self.done[idx], dtype=torch.float32, device=device),
            "weights": torch.tensor(weights.reshape(-1, 1), dtype=torch.float32, device=device),
            "idx": idx,
        }

    def update_priorities(self, idx, td_errors: np.ndarray):
        new_priorities = np.abs(td_errors).reshape(-1) + self.eps
        self.priorities[idx] = new_priorities.astype(np.float32)
        self.max_priority = max(self.max_priority, float(new_priorities.max()))

    def priority_stats(self) -> Tuple[float, float]:
        if self.size == 0:
            return 0.0, 0.0
        p = self.priorities[:self.size]
        return float(np.mean(p)), float(np.max(p))


@dataclass
class PrioritizedTrainStats(TrainStats):
    td_error_mean: float
    td_error_max: float
    importance_weight_mean: float
    td_errors: np.ndarray


class PrioritizedSACAgent:
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

    def update(self, batch) -> PrioritizedTrainStats:
        obs, acts, rews, next_obs, done = batch["obs"], batch["acts"], batch["rews"], batch["next_obs"], batch["done"]
        weights = batch["weights"]

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            y = rews + self.gamma * (1.0 - done) * target_q

        q1 = self.critic1(obs, acts)
        q2 = self.critic2(obs, acts)
        td1 = q1 - y
        td2 = q2 - y
        critic1_loss = (weights * td1.pow(2)).mean()
        critic2_loss = (weights * td2.pow(2)).mean()
        critic_loss = critic1_loss + critic2_loss

        self.critic1_opt.zero_grad(); self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step(); self.critic2_opt.step()

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

        td_errors = 0.5 * (td1.detach().abs() + td2.detach().abs())
        return PrioritizedTrainStats(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            q_mean=float(q1.mean().item()),
            alpha=float(self.alpha.detach().item()),
            td_error_mean=float(td_errors.mean().item()),
            td_error_max=float(td_errors.max().item()),
            importance_weight_mean=float(weights.mean().item()),
            td_errors=td_errors.cpu().numpy(),
        )

    def soft_update(self, net: nn.Module, target_net: nn.Module) -> None:
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


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


def summarize_run(csv_path: str, algorithm: str, args, model_path: str, last_n: int = 5) -> dict:
    df = pd.read_csv(csv_path)
    eval_success = df["eval_success"].dropna() if "eval_success" in df.columns else pd.Series(dtype=float)
    eval_return = df["eval_return"].dropna() if "eval_return" in df.columns else pd.Series(dtype=float)
    eval_dist = df["eval_true_goal_distance"].dropna() if "eval_true_goal_distance" in df.columns else pd.Series(dtype=float)
    return {
        "algorithm": algorithm, "task": args.task, "reward_type": args.reward_type, "seed": args.seed,
        "episodes": args.episodes, "use_her": bool(args.use_her), "her_k": args.her_k,
        "per_alpha": args.per_alpha, "per_beta_start": args.per_beta_start, "per_beta_end": args.per_beta_end,
        "final_eval_success": float(eval_success.iloc[-1]) if len(eval_success) else float("nan"),
        "best_eval_success": float(eval_success.max()) if len(eval_success) else float("nan"),
        "last5_eval_success_mean": float(eval_success.tail(last_n).mean()) if len(eval_success) else float("nan"),
        "final_eval_return": float(eval_return.iloc[-1]) if len(eval_return) else float("nan"),
        "last5_eval_distance_mean": float(eval_dist.tail(last_n).mean()) if len(eval_dist) else float("nan"),
        "csv_path": csv_path, "model_path": model_path,
    }


def write_summary_files(summary: dict, run_summary_dir: str, global_results_dir: str) -> None:
    pd.DataFrame([summary]).to_csv(os.path.join(run_summary_dir, "summary.csv"), index=False)
    with open(os.path.join(run_summary_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    global_csv = os.path.join(global_results_dir, "prioritized_her_results.csv")
    new_row = pd.DataFrame([summary])
    if os.path.exists(global_csv):
        old = pd.read_csv(global_csv)
        key = (old["algorithm"] == summary["algorithm"]) & (old["task"] == summary["task"]) & (old["reward_type"] == summary["reward_type"]) & (old["seed"] == summary["seed"])
        old = old.loc[~key]
        out = pd.concat([old, new_row], ignore_index=True)
    else:
        out = new_row
    out.sort_values(["task", "reward_type", "algorithm", "seed"]).to_csv(global_csv, index=False)


def linear_beta(episode_idx: int, episodes: int, beta_start: float, beta_end: float) -> float:
    frac = min(1.0, max(0.0, episode_idx / max(1, episodes)))
    return beta_start + frac * (beta_end - beta_start)


def main():
    parser = argparse.ArgumentParser(description="Manual SAC + prioritized HER for Fetch tasks.")
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
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-on-first-episode", action="store_true")
    parser.add_argument("--carry-eval-forward", action="store_true")
    parser.add_argument("--plot-smooth", type=int, default=10)
    parser.add_argument("--prioritized-her-dir", type=str, default="prioritized_her")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    algorithm = "sac_prioritized_her" if args.use_her else "sac_prioritized_replay"

    env = make_env(args.task, args.reward_type, args.seed)
    obs_dict, _ = env.reset(seed=args.seed)
    obs = flatten_obs(obs_dict)
    obs_dim, act_dim = obs.shape[0], env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = PrioritizedSACAgent(obs_dim, act_dim, act_limit, args.hidden_dim, args.actor_lr, args.critic_lr, args.gamma, args.tau, device,
                                alpha=args.alpha, auto_alpha=args.auto_alpha, alpha_lr=args.alpha_lr)
    replay = PrioritizedReplayBuffer(obs_dim, act_dim, args.replay_size, alpha=args.per_alpha)

    run_dirs = create_run_dirs(args.prioritized_her_dir, algorithm, args.task, args.reward_type, args.seed)
    run_name = f"{algorithm}_{args.task}_{args.reward_type}_seed{args.seed}"
    csv_path = os.path.join(run_dirs["logs"], f"{run_name}.csv")
    model_path = os.path.join(run_dirs["models"], f"{run_name}_final.pt")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_steps", "train_return", "train_true_success", "train_true_goal_distance",
                         "eval_return", "eval_success", "eval_true_goal_distance", "critic_loss", "actor_loss", "q_mean", "alpha",
                         "td_error_mean", "td_error_max", "importance_weight_mean", "priority_mean", "priority_max",
                         "per_beta", "her_transitions_added", "replay_size"])

        total_steps = 0
        recent_success = deque(maxlen=10)
        last_eval_return = last_eval_success = last_eval_distance = np.nan

        for episode_idx in range(1, args.episodes + 1):
            obs_dict, _ = env.reset()
            true_goal = obs_dict["desired_goal"].copy()
            done = truncated = False
            ep_return = 0.0
            ep_success = 0.0
            ep_distance = np.nan
            episode_transitions = []
            episode_steps = 0
            critic_losses = []; actor_losses = []; q_means = []; alphas = []; td_means = []; td_maxes = []; iw_means = []

            while not (done or truncated):
                obs = flatten_obs(obs_dict)
                action = env.action_space.sample() if total_steps < args.start_steps else agent.select_action(obs, deterministic=False)
                next_obs_dict, reward, done, truncated, info = env.step(action)
                terminal = float(done)
                episode_transitions.append(make_transition_from_obs_dict(obs_dict, action, reward, next_obs_dict, terminal, info))
                ep_return += float(reward)
                ep_success = max(ep_success, float(info.get("is_success", 0.0)))
                ep_distance = goal_distance(next_obs_dict["achieved_goal"], true_goal)
                obs_dict = next_obs_dict
                total_steps += 1
                episode_steps += 1

            her_added = add_episode_with_her(replay, episode_transitions, lambda ag, dg, info: env_compute_reward(env, ag, dg, info), args.her_k, args.use_her, args.her_future_offset)
            beta = linear_beta(episode_idx, args.episodes, args.per_beta_start, args.per_beta_end)

            if replay.size >= args.batch_size:
                for _ in range(max(1, episode_steps * args.updates_per_step)):
                    batch = replay.sample(args.batch_size, device, beta=beta)
                    stats = agent.update(batch)
                    replay.update_priorities(batch["idx"], stats.td_errors)
                    critic_losses.append(stats.critic_loss); actor_losses.append(stats.actor_loss); q_means.append(stats.q_mean); alphas.append(stats.alpha)
                    td_means.append(stats.td_error_mean); td_maxes.append(stats.td_error_max); iw_means.append(stats.importance_weight_mean)

            priority_mean, priority_max = replay.priority_stats()
            recent_success.append(ep_success)
            should_eval = (episode_idx % args.eval_every == 0) or (episode_idx == 1 and args.eval_on_first_episode)
            if should_eval:
                last_eval_return, last_eval_success, last_eval_distance = evaluate(agent, args.task, args.reward_type, args.seed, args.eval_episodes)
            eval_return = last_eval_return if (should_eval or args.carry_eval_forward) else np.nan
            eval_success = last_eval_success if (should_eval or args.carry_eval_forward) else np.nan
            eval_distance = last_eval_distance if (should_eval or args.carry_eval_forward) else np.nan

            writer.writerow([episode_idx, total_steps, ep_return, ep_success, ep_distance, eval_return, eval_success, eval_distance,
                             float(np.mean(critic_losses)) if critic_losses else np.nan, float(np.mean(actor_losses)) if actor_losses else np.nan,
                             float(np.mean(q_means)) if q_means else np.nan, float(np.mean(alphas)) if alphas else float(agent.alpha.detach().item()),
                             float(np.mean(td_means)) if td_means else np.nan, float(np.max(td_maxes)) if td_maxes else np.nan,
                             float(np.mean(iw_means)) if iw_means else np.nan, priority_mean, priority_max, beta, her_added, replay.size])
            f.flush()

            print(f"Episode {episode_idx:04d} | algo={algorithm} | task={args.task} | reward={args.reward_type} | return={ep_return:.2f} | "
                  f"train_success={ep_success:.1f} (recent={float(np.mean(recent_success)):.3f}) | dist={ep_distance:.4f} | "
                  f"eval_success={(eval_success if not np.isnan(eval_success) else float('nan')):.3f} | TD={float(np.mean(td_means)) if td_means else float('nan'):.4f} | HER+={her_added} | replay={replay.size}")

    env.close()
    torch.save({"algorithm": algorithm, "args": vars(args), "actor_state_dict": agent.actor.state_dict()}, model_path)
    summary = summarize_run(csv_path, algorithm, args, model_path)
    write_summary_files(summary, run_dirs["summaries"], run_dirs["results"])
    print(f"Saved log to: {csv_path}")
    print(f"Saved model to: {model_path}")
    plot_from_csv(csv_path, outdir=run_dirs["plots"], smooth=args.plot_smooth)


if __name__ == "__main__":
    main()
