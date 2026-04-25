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
from typing import Dict, Tuple

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F



from her.her import (
    add_episode_with_her,
    env_compute_reward,
    flatten_goal_obs,
    make_transition_from_obs_dict,
)
from plot_all_results import plot_from_csv

gym.register_envs(gymnasium_robotics)

FETCH_SUCCESS_THRESHOLD = 0.05
LOG_STD_MIN = -20
LOG_STD_MAX = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_obs(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    return flatten_goal_obs(
        obs_dict["observation"],
        obs_dict["achieved_goal"],
        obs_dict["desired_goal"],
    )


def check_true_goal_success(achieved_goal: np.ndarray, true_goal: np.ndarray) -> float:
    return float(np.linalg.norm(achieved_goal - true_goal) < FETCH_SUCCESS_THRESHOLD)


def goal_distance(achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
    return float(np.linalg.norm(achieved_goal - desired_goal))


@dataclass
class CurriculumState:
    ratio: float


def set_env_goal(env, goal: np.ndarray) -> None:
    """Set Fetch environment goal for the current episode."""
    env.unwrapped.goal = goal.copy()


def curriculum_ratio_for_episode(args, episode_idx: int, state: CurriculumState, recent_curr_success) -> float:
    if not args.use_curriculum:
        return 1.0
    if args.curriculum_mode == "linear":
        if episode_idx <= args.curriculum_warmup_episodes:
            return float(args.curriculum_start_ratio)
        denom = max(1, args.curriculum_duration_episodes)
        progress = min(1.0, (episode_idx - args.curriculum_warmup_episodes) / denom)
        return float(args.curriculum_start_ratio + progress * (args.curriculum_end_ratio - args.curriculum_start_ratio))
    if args.curriculum_mode == "adaptive":
        if len(recent_curr_success) >= args.curriculum_window:
            mean_success = float(np.mean(recent_curr_success))
            if mean_success >= args.curriculum_success_threshold:
                state.ratio = min(args.curriculum_end_ratio, state.ratio + args.curriculum_step_up)
            elif mean_success <= args.curriculum_failure_threshold:
                state.ratio = max(args.curriculum_start_ratio, state.ratio - args.curriculum_step_down)
        return float(state.ratio)
    raise ValueError(f"Unknown curriculum mode: {args.curriculum_mode}")


def make_curriculum_goal(start_goal: np.ndarray, true_goal: np.ndarray, ratio: float) -> np.ndarray:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return (start_goal + ratio * (true_goal - start_goal)).astype(np.float32)


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
        if self.size < batch_size:
            raise ValueError(f"Replay buffer has {self.size} transitions, needs {batch_size}.")
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32, device=device),
            "acts": torch.tensor(self.acts[idx], dtype=torch.float32, device=device),
            "rews": torch.tensor(self.rews[idx], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "done": torch.tensor(self.done[idx], dtype=torch.float32, device=device),
        }


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, act_limit: float):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, torch.exp(log_std)

    def sample(self, obs: torch.Tensor):
        mean, std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.act_limit
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.act_limit * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.act_limit
        return action, log_prob, mean_action


@dataclass
class TrainStats:
    critic_loss: float
    actor_loss: float
    q_mean: float
    alpha: float


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
        obs = batch["obs"]
        acts = batch["acts"]
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            y = rews + self.gamma * (1.0 - done) * target_q

        q1 = self.critic1(obs, acts)
        q2 = self.critic2(obs, acts)
        critic1_loss = F.mse_loss(q1, y)
        critic2_loss = F.mse_loss(q2, y)
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

    def soft_update(self, net: nn.Module, target_net: nn.Module) -> None:
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


def make_env(task: str, reward_type: str, seed: int):
    env = gym.make(f"{task}-v4", reward_type=reward_type)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


@torch.no_grad()
def evaluate(agent: SACAgent, task: str, reward_type: str, seed: int, episodes: int) -> Tuple[float, float, float]:
    env = make_env(task, reward_type, seed + 10_000)
    returns, successes, distances = [], [], []
    for ep in range(episodes):
        obs_dict, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        truncated = False
        ep_return = 0.0
        ep_success = 0.0
        final_distance = np.nan
        while not (done or truncated):
            obs = flatten_obs(obs_dict)
            act = agent.select_action(obs, deterministic=True)
            obs_dict, rew, done, truncated, info = env.step(act)
            ep_return += float(rew)
            ep_success = max(ep_success, float(info.get("is_success", 0.0)))
            final_distance = goal_distance(obs_dict["achieved_goal"], obs_dict["desired_goal"])
        returns.append(ep_return)
        successes.append(ep_success)
        distances.append(final_distance)
    env.close()
    return float(np.mean(returns)), float(np.mean(successes)), float(np.mean(distances))


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


def summarize_run(csv_path: str, algorithm: str, task: str, reward_type: str, seed: int, args, model_path: str, last_n: int = 5) -> dict:
    df = pd.read_csv(csv_path)
    eval_success = df["eval_success"].dropna() if "eval_success" in df.columns else pd.Series(dtype=float)
    eval_return = df["eval_return"].dropna() if "eval_return" in df.columns else pd.Series(dtype=float)
    eval_dist = df["eval_true_goal_distance"].dropna() if "eval_true_goal_distance" in df.columns else pd.Series(dtype=float)
    her_added = df["her_transitions_added"].dropna() if "her_transitions_added" in df.columns else pd.Series(dtype=float)
    return {
        "algorithm": algorithm,
        "task": task,
        "reward_type": reward_type,
        "seed": seed,
        "episodes": args.episodes,
        "use_her": bool(args.use_her),
        "her_k": args.her_k,
        "future_offset": args.her_future_offset,
        "use_curriculum": bool(args.use_curriculum),
        "curriculum_mode": args.curriculum_mode,
        "curriculum_start_ratio": args.curriculum_start_ratio,
        "curriculum_end_ratio": args.curriculum_end_ratio,
        "curriculum_duration_episodes": args.curriculum_duration_episodes,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "final_eval_success": float(eval_success.iloc[-1]) if len(eval_success) else float("nan"),
        "best_eval_success": float(eval_success.max()) if len(eval_success) else float("nan"),
        "last5_eval_success_mean": float(eval_success.tail(last_n).mean()) if len(eval_success) else float("nan"),
        "last5_eval_success_std": float(eval_success.tail(last_n).std(ddof=0)) if len(eval_success) else float("nan"),
        "final_eval_return": float(eval_return.iloc[-1]) if len(eval_return) else float("nan"),
        "best_eval_return": float(eval_return.max()) if len(eval_return) else float("nan"),
        "last5_eval_return_mean": float(eval_return.tail(last_n).mean()) if len(eval_return) else float("nan"),
        "last5_eval_distance_mean": float(eval_dist.tail(last_n).mean()) if len(eval_dist) else float("nan"),
        "total_her_transitions_added": float(her_added.sum()) if len(her_added) else 0.0,
        "csv_path": csv_path,
        "model_path": model_path,
    }


def write_summary_files(summary: dict, run_summary_dir: str, global_results_dir: str) -> None:
    pd.DataFrame([summary]).to_csv(os.path.join(run_summary_dir, "summary.csv"), index=False)
    with open(os.path.join(run_summary_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    global_csv = os.path.join(global_results_dir, "her_results.csv")
    new_row = pd.DataFrame([summary])
    if os.path.exists(global_csv):
        old = pd.read_csv(global_csv)
        key = (
            (old["algorithm"] == summary["algorithm"]) &
            (old["task"] == summary["task"]) &
            (old["reward_type"] == summary["reward_type"]) &
            (old["seed"] == summary["seed"])
        )
        old = old.loc[~key]
        out = pd.concat([old, new_row], ignore_index=True)
    else:
        out = new_row
    out.sort_values(["task", "reward_type", "algorithm", "seed"]).to_csv(global_csv, index=False)


def save_checkpoint(agent: SACAgent, model_path: str, algorithm: str, args, obs_dim: int, act_dim: int, act_limit: float) -> None:
    checkpoint = {
        "algorithm": algorithm,
        "task": args.task,
        "reward_type": args.reward_type,
        "seed": args.seed,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "act_limit": act_limit,
        "hidden_dim": args.hidden_dim,
        "args": vars(args),
        "actor_state_dict": agent.actor.state_dict(),
    }
    torch.save(checkpoint, model_path)


def main():
    parser = argparse.ArgumentParser(description="Manual SAC + HER for Gymnasium-Robotics Fetch tasks.")
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
    parser.add_argument("--her-future-offset", type=int, default=1, help="1=strictly future goals when possible; 0=current/future.")
    parser.add_argument("--use-curriculum", action="store_true", help="Train on intermediate goals that gradually move toward the true goal.")
    parser.add_argument("--curriculum-mode", type=str, default="linear", choices=["linear", "adaptive"])
    parser.add_argument("--curriculum-start-ratio", type=float, default=0.15, help="0=start at achieved_goal, 1=original desired_goal.")
    parser.add_argument("--curriculum-end-ratio", type=float, default=1.0)
    parser.add_argument("--curriculum-warmup-episodes", type=int, default=0)
    parser.add_argument("--curriculum-duration-episodes", type=int, default=700)
    parser.add_argument("--curriculum-window", type=int, default=20)
    parser.add_argument("--curriculum-success-threshold", type=float, default=0.70)
    parser.add_argument("--curriculum-failure-threshold", type=float, default=0.20)
    parser.add_argument("--curriculum-step-up", type=float, default=0.05)
    parser.add_argument("--curriculum-step-down", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-on-first-episode", action="store_true")
    parser.add_argument("--carry-eval-forward", action="store_true")
    parser.add_argument("--plot-smooth", type=int, default=10)
    parser.add_argument("--her-dir", type=str, default="her")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    algorithm = ("sac_her_curriculum" if args.use_her else "sac_curriculum") if args.use_curriculum else ("sac_her" if args.use_her else "sac_no_her")

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
    curriculum_state = CurriculumState(ratio=float(args.curriculum_start_ratio))
    recent_curr_success = deque(maxlen=args.curriculum_window)

    run_dirs = create_run_dirs(args.her_dir, algorithm, args.task, args.reward_type, args.seed)
    run_name = f"{algorithm}_{args.task}_{args.reward_type}_seed{args.seed}"
    csv_path = os.path.join(run_dirs["logs"], f"{run_name}.csv")
    model_path = os.path.join(run_dirs["models"], f"{run_name}_final.pt")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "total_steps", "curriculum_ratio", "train_return",
            "train_curriculum_success", "train_true_success",
            "train_curriculum_goal_distance", "train_true_goal_distance",
            "eval_return", "eval_success", "eval_true_goal_distance",
            "critic_loss", "actor_loss", "q_mean", "alpha", "her_transitions_added", "replay_size",
        ])

        total_steps = 0
        recent_success = deque(maxlen=10)
        recent_true_success = deque(maxlen=10)
        last_eval_return = np.nan
        last_eval_success = np.nan
        last_eval_distance = np.nan

        for episode_idx in range(1, args.episodes + 1):
            obs_dict, _ = env.reset()
            true_goal = obs_dict["desired_goal"].copy()
            start_goal = obs_dict["achieved_goal"].copy()
            curriculum_ratio = curriculum_ratio_for_episode(args, episode_idx, curriculum_state, recent_curr_success)
            train_goal = make_curriculum_goal(start_goal, true_goal, curriculum_ratio) if args.use_curriculum else true_goal.copy()
            set_env_goal(env, train_goal)
            obs_dict["desired_goal"] = train_goal.copy()
            done = False
            truncated = False
            ep_return = 0.0
            ep_curr_success = 0.0
            ep_true_success = 0.0
            ep_curr_distance = np.nan
            ep_true_distance = np.nan
            episode_transitions = []
            episode_steps = 0
            critic_losses, actor_losses, q_means, alphas = [], [], [], []

            while not (done or truncated):
                obs = flatten_obs(obs_dict)
                if total_steps < args.start_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs, deterministic=False)

                next_obs_dict, reward, done, truncated, info = env.step(action)
                next_obs_dict["desired_goal"] = train_goal.copy()
                terminal = float(done)  # Fetch usually ends by truncation/time-limit; do not mark truncated as terminal.
                episode_transitions.append(
                    make_transition_from_obs_dict(obs_dict, action, reward, next_obs_dict, terminal, info)
                )

                ep_return += float(reward)
                ep_curr_success = max(ep_curr_success, float(info.get("is_success", 0.0)))
                ep_true_success = max(ep_true_success, check_true_goal_success(next_obs_dict["achieved_goal"], true_goal))
                ep_curr_distance = goal_distance(next_obs_dict["achieved_goal"], train_goal)
                ep_true_distance = goal_distance(next_obs_dict["achieved_goal"], true_goal)

                obs_dict = next_obs_dict
                total_steps += 1
                episode_steps += 1

            her_added = add_episode_with_her(
                replay,
                episode_transitions,
                reward_fn=lambda ag, dg, info: env_compute_reward(env, ag, dg, info),
                her_k=args.her_k,
                use_her=args.use_her,
                future_offset=args.her_future_offset,
            )

            if replay.size >= args.batch_size:
                for _ in range(max(1, episode_steps * args.updates_per_step)):
                    batch = replay.sample(args.batch_size, device)
                    stats = agent.update(batch)
                    critic_losses.append(stats.critic_loss)
                    actor_losses.append(stats.actor_loss)
                    q_means.append(stats.q_mean)
                    alphas.append(stats.alpha)

            recent_success.append(ep_curr_success)
            recent_true_success.append(ep_true_success)
            recent_curr_success.append(ep_curr_success)
            should_eval = (episode_idx % args.eval_every == 0) or (episode_idx == 1 and args.eval_on_first_episode)
            if should_eval:
                last_eval_return, last_eval_success, last_eval_distance = evaluate(
                    agent, args.task, args.reward_type, args.seed, args.eval_episodes
                )

            if args.carry_eval_forward:
                eval_return, eval_success, eval_distance = last_eval_return, last_eval_success, last_eval_distance
            else:
                eval_return = last_eval_return if should_eval else np.nan
                eval_success = last_eval_success if should_eval else np.nan
                eval_distance = last_eval_distance if should_eval else np.nan

            mean_critic_loss = float(np.mean(critic_losses)) if critic_losses else np.nan
            mean_actor_loss = float(np.mean(actor_losses)) if actor_losses else np.nan
            mean_q = float(np.mean(q_means)) if q_means else np.nan
            mean_alpha = float(np.mean(alphas)) if alphas else float(agent.alpha.detach().item())

            writer.writerow([
                episode_idx, total_steps, curriculum_ratio, ep_return,
                ep_curr_success, ep_true_success, ep_curr_distance, ep_true_distance,
                eval_return, eval_success, eval_distance,
                mean_critic_loss, mean_actor_loss, mean_q, mean_alpha, her_added, replay.size,
            ])
            f.flush()

            eval_success_str = f"{eval_success:.3f}" if not np.isnan(eval_success) else "NA"
            recent_success_str = f"{float(np.mean(recent_success)):.3f}" if recent_success else "NA"
            recent_true_success_str = f"{float(np.mean(recent_true_success)):.3f}" if recent_true_success else "NA"
            print(
                f"Episode {episode_idx:04d} | algo={algorithm} | task={args.task} | reward={args.reward_type} | "
                f"return={ep_return:.2f} | cur_ratio={curriculum_ratio:.2f} | "
                f"curr_success={ep_curr_success:.1f} (recent={recent_success_str}) | "
                f"true_success={ep_true_success:.1f} (recent={recent_true_success_str}) | "
                f"curr_dist={ep_curr_distance:.4f} | true_dist={ep_true_distance:.4f} | eval_success={eval_success_str} | "
                f"HER+={her_added} | replay={replay.size} | critic_loss={mean_critic_loss:.6f} | "
                f"actor_loss={mean_actor_loss:.6f} | q_mean={mean_q:.6f} | alpha={mean_alpha:.4f}"
            )

    env.close()
    save_checkpoint(agent, model_path, algorithm, args, obs_dim, act_dim, act_limit)
    summary = summarize_run(csv_path, algorithm, args.task, args.reward_type, args.seed, args, model_path)
    write_summary_files(summary, run_dirs["summaries"], run_dirs["results"])

    print(f"Saved log to: {csv_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved run summary to: {os.path.join(run_dirs['summaries'], 'summary.csv')}")
    print(f"Updated global table: {os.path.join(run_dirs['results'], 'her_results.csv')}")
    plot_from_csv(csv_path, outdir=run_dirs["plots"], smooth=args.plot_smooth)


if __name__ == "__main__":
    main()
