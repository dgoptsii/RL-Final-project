"""
Train SAC with optional HER and optional simplified HIRO-style hierarchy on Fetch tasks.

Important corrections in this version:
1. HER rewards are recomputed with env.compute_reward(...), not a hand-written formula.
2. Non-HER training also uses env.compute_reward for the current low-level goal.
3. Checkpoints are saved after evaluation when eval_success improves.
4. Learned-HIRO high-level replay segment handling is fixed; heuristic HIRO remains recommended first.

Examples:
  python train_sac_hiro_her.py --task FetchPush --reward-type sparse --use-her
  python train_sac_hiro_her.py --task FetchPush --reward-type sparse --use-her --use-hiro
"""
from __future__ import annotations

import argparse
import csv
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
import torch.nn as nn
import torch.nn.functional as F

from checkpoints.checkpoint_utils import save_training_checkpoint
from her.her import EpisodeTransition, add_episode_with_her, env_compute_reward, flatten_goal_obs
from hiro import (
    HighLevelReplayBuffer,
    HighLevelTD3Agent,
    goal_distance,
    heuristic_subgoal,
    subgoal_success,
)
from plot.plot_hiro_her_results import plot_from_csv

gym.register_envs(gymnasium_robotics)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(task: str, reward_type: str, seed: int, render_mode: Optional[str] = None):
    env_name = f"{task}-v4"
    kwargs = {"reward_type": reward_type}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(env_name, **kwargs)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def flatten_obs_dict(obs_dict: Dict) -> np.ndarray:
    return flatten_goal_obs(obs_dict["observation"], obs_dict["achieved_goal"], obs_dict["desired_goal"])


def make_low_level_obs(obs_dict: Dict, goal: np.ndarray) -> np.ndarray:
    return flatten_goal_obs(obs_dict["observation"], obs_dict["achieved_goal"], goal)


def make_high_level_obs(obs_dict: Dict) -> np.ndarray:
    return flatten_obs_dict(obs_dict)


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


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, act_limit: float):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
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
        log_prob -= torch.log(self.act_limit * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.act_limit
        return action, log_prob, mean_action


@dataclass
class TrainStats:
    critic_loss: float
    actor_loss: float
    q_mean: float


class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit, hidden_dim, actor_lr, critic_lr, gamma, tau, device, alpha=0.2):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit
        self.alpha = alpha
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

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, stochastic: bool = True) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, mean_action = self.actor.sample(obs_t)
        chosen = action if stochastic else mean_action
        return np.clip(chosen.cpu().numpy()[0], -self.act_limit, self.act_limit)

    def update(self, batch) -> TrainStats:
        obs, acts, rews, next_obs, done = batch["obs"], batch["acts"], batch["rews"], batch["next_obs"], batch["done"]
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            y = rews + self.gamma * (1.0 - done) * target_q

        q1 = self.critic1(obs, acts)
        q2 = self.critic2(obs, acts)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic1_opt.zero_grad(); self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step(); self.critic2_opt.step()

        new_actions, log_prob, _ = self.actor.sample(obs)
        q_new = torch.min(self.critic1(obs, new_actions), self.critic2(obs, new_actions))
        actor_loss = (self.alpha * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        return TrainStats(float(critic_loss.item()), float(actor_loss.item()), float(q1.mean().item()))

    def soft_update(self, net, target_net):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


def select_hiro_subgoal(args, obs_dict, high_agent, total_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    achieved = obs_dict["achieved_goal"].copy()
    desired = obs_dict["desired_goal"].copy()
    if args.hiro_high_level_mode == "heuristic":
        subgoal = heuristic_subgoal(achieved, desired, ratio=args.hiro_subgoal_ratio, noise_std=args.hiro_subgoal_noise)
        return subgoal.astype(np.float32), (subgoal - achieved).astype(np.float32)

    high_obs = make_high_level_obs(obs_dict)
    if total_steps < args.hiro_high_start_steps:
        delta = np.random.uniform(-args.hiro_subgoal_limit, args.hiro_subgoal_limit, size=achieved.shape).astype(np.float32)
    else:
        delta = high_agent.select_subgoal_delta(high_obs, noise_std=args.hiro_high_noise)
    subgoal = achieved + delta
    return subgoal.astype(np.float32), delta.astype(np.float32)


@torch.no_grad()
def evaluate(agent, args, seed: int, episodes: int, high_agent=None) -> Tuple[float, float, float, float, float]:
    env = make_env(args.task, args.reward_type, seed + 10_000)
    returns, successes, true_distances, subgoal_distances, subgoal_successes = [], [], [], [], []

    for ep in range(episodes):
        obs_dict, _ = env.reset(seed=seed + 10_000 + ep)
        done = truncated = False
        ep_return = 0.0
        ep_success = 0.0
        ep_subgoal_success = 0.0
        subgoal = obs_dict["desired_goal"].copy()
        steps_since_high = args.hiro_interval

        while not (done or truncated):
            if args.use_hiro and steps_since_high >= args.hiro_interval:
                subgoal, _ = select_hiro_subgoal(args, obs_dict, high_agent, total_steps=10**9)
                steps_since_high = 0

            goal = subgoal if args.use_hiro else obs_dict["desired_goal"]
            low_obs = make_low_level_obs(obs_dict, goal)
            action = agent.select_action(low_obs, stochastic=False)
            next_obs_dict, env_rew, done, truncated, info = env.step(action)
            ep_return += env_rew
            ep_success = max(ep_success, float(info.get("is_success", 0.0)))
            if args.use_hiro:
                ep_subgoal_success = max(ep_subgoal_success, subgoal_success(next_obs_dict["achieved_goal"], subgoal))
            obs_dict = next_obs_dict
            steps_since_high += 1

        returns.append(ep_return)
        successes.append(ep_success)
        true_distances.append(goal_distance(obs_dict["achieved_goal"], obs_dict["desired_goal"]))
        if args.use_hiro:
            subgoal_distances.append(goal_distance(obs_dict["achieved_goal"], subgoal))
            subgoal_successes.append(ep_subgoal_success)
        else:
            subgoal_distances.append(np.nan)
            subgoal_successes.append(np.nan)

    env.close()
    return (
        float(np.mean(returns)),
        float(np.mean(successes)),
        float(np.nanmean(subgoal_successes)) if np.any(~np.isnan(subgoal_successes)) else np.nan,
        float(np.mean(true_distances)),
        float(np.nanmean(subgoal_distances)) if np.any(~np.isnan(subgoal_distances)) else np.nan,
    )


def update_many(agent, replay, args, device, losses, actor_losses, q_means):
    if replay.size >= args.batch_size:
        for _ in range(args.updates_per_step):
            stats = agent.update(replay.sample(args.batch_size, device))
            losses.append(stats.critic_loss)
            actor_losses.append(stats.actor_loss)
            q_means.append(stats.q_mean)


def parse_args():
    parser = argparse.ArgumentParser(description="SAC with optional HER and simplified HIRO for Fetch tasks.")
    parser.add_argument("--task", choices=["FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace"], default="FetchReach")
    parser.add_argument("--reward-type", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--updates-per-step", type=int, default=1)

    parser.add_argument("--use-her", action="store_true")
    parser.add_argument("--her-k", type=int, default=4)
    parser.add_argument("--her-future-offset", type=int, default=0)

    parser.add_argument("--use-hiro", action="store_true")
    parser.add_argument("--hiro-interval", type=int, default=10)
    parser.add_argument("--hiro-high-level-mode", choices=["heuristic", "learned"], default="heuristic")
    parser.add_argument("--hiro-subgoal-ratio", type=float, default=0.5)
    parser.add_argument("--hiro-subgoal-noise", type=float, default=0.01)
    parser.add_argument("--hiro-subgoal-limit", type=float, default=0.25)
    parser.add_argument("--hiro-high-hidden-dim", type=int, default=256)
    parser.add_argument("--hiro-high-actor-lr", type=float, default=1e-4)
    parser.add_argument("--hiro-high-critic-lr", type=float, default=3e-4)
    parser.add_argument("--hiro-high-gamma", type=float, default=0.98)
    parser.add_argument("--hiro-high-tau", type=float, default=0.005)
    parser.add_argument("--hiro-high-noise", type=float, default=0.05)
    parser.add_argument("--hiro-high-start-steps", type=int, default=1000)
    parser.add_argument("--hiro-high-replay-size", type=int, default=50000)
    parser.add_argument("--hiro-high-batch-size", type=int, default=128)
    parser.add_argument("--hiro-high-updates", type=int, default=1)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-on-first-episode", action="store_true")
    parser.add_argument("--carry-eval-forward", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs_hiro_her")
    parser.add_argument("--plots-dir", type=str, default="plots_hiro_her")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_hiro_her")
    parser.add_argument("--plot-smooth", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    env = make_env(args.task, args.reward_type, args.seed)
    obs_dict, _ = env.reset(seed=args.seed)
    low_obs = make_low_level_obs(obs_dict, obs_dict["desired_goal"])
    low_obs_dim = low_obs.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    subgoal_dim = obs_dict["achieved_goal"].shape[0]
    high_obs_dim = flatten_obs_dict(obs_dict).shape[0]

    agent = SACAgent(low_obs_dim, act_dim, act_limit, args.hidden_dim, args.actor_lr, args.critic_lr, args.gamma, args.tau, device, args.alpha)
    replay = ReplayBuffer(low_obs_dim, act_dim, args.replay_size)

    high_agent = None
    high_replay = None
    if args.use_hiro and args.hiro_high_level_mode == "learned":
        high_agent = HighLevelTD3Agent(
            obs_dim=high_obs_dim,
            subgoal_dim=subgoal_dim,
            hidden_dim=args.hiro_high_hidden_dim,
            subgoal_limit=args.hiro_subgoal_limit,
            actor_lr=args.hiro_high_actor_lr,
            critic_lr=args.hiro_high_critic_lr,
            gamma=args.hiro_high_gamma,
            tau=args.hiro_high_tau,
            device=device,
        )
        high_replay = HighLevelReplayBuffer(high_obs_dim, subgoal_dim, args.hiro_high_replay_size)

    method = "sac"
    if args.use_her:
        method += "_her"
    if args.use_hiro:
        method += "_hiro"
    csv_path = os.path.join(args.log_dir, f"{method}_{args.task}_{args.reward_type}_seed{args.seed}.csv")
    best_eval_success = -np.inf

    reward_fn = lambda achieved, desired, info=None: env_compute_reward(env, achieved, desired, info)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "total_steps", "train_return", "train_true_success", "train_subgoal_success",
            "train_true_goal_distance", "train_subgoal_distance", "eval_return", "eval_success",
            "eval_subgoal_success", "eval_true_goal_distance", "eval_subgoal_distance",
            "critic_loss", "actor_loss", "q_mean", "high_critic_loss", "high_actor_loss", "high_q_mean",
            "her_transitions_added", "replay_size", "high_replay_size"
        ])

        total_steps = 0
        last_eval = (np.nan, np.nan, np.nan, np.nan, np.nan)
        recent_eval_success = deque(maxlen=5)

        for episode in range(1, args.episodes + 1):
            obs_dict, _ = env.reset()
            done = truncated = False
            ep_return = 0.0
            ep_true_success = 0.0
            ep_subgoal_success = 0.0
            true_distances = []
            subgoal_distances = []
            critic_losses, actor_losses, q_means = [], [], []
            high_critic_losses, high_actor_losses, high_q_means = [], [], []
            episode_transitions: List[EpisodeTransition] = []
            her_added = 0

            current_subgoal = obs_dict["desired_goal"].copy()
            current_delta = current_subgoal - obs_dict["achieved_goal"]
            high_start_obs = make_high_level_obs(obs_dict)
            high_reward_accum = 0.0
            steps_since_high = args.hiro_interval
            have_pending_high_segment = False

            while not (done or truncated):
                if args.use_hiro and steps_since_high >= args.hiro_interval:
                    if high_replay is not None and have_pending_high_segment:
                        high_replay.add(high_start_obs, current_delta, high_reward_accum, make_high_level_obs(obs_dict), float(done))
                    current_subgoal, current_delta = select_hiro_subgoal(args, obs_dict, high_agent, total_steps)
                    high_start_obs = make_high_level_obs(obs_dict)
                    high_reward_accum = 0.0
                    steps_since_high = 0
                    have_pending_high_segment = True

                low_goal = current_subgoal if args.use_hiro else obs_dict["desired_goal"]
                low_obs = make_low_level_obs(obs_dict, low_goal)
                if total_steps < args.start_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(low_obs, stochastic=True)

                next_obs_dict, env_reward, done, truncated, info = env.step(action)

                low_reward = reward_fn(next_obs_dict["achieved_goal"], low_goal, info)
                low_next_obs = make_low_level_obs(next_obs_dict, low_goal)

                if args.use_her:
                    episode_transitions.append(
                        EpisodeTransition(
                            observation=obs_dict["observation"].copy(),
                            achieved_goal=obs_dict["achieved_goal"].copy(),
                            desired_goal=low_goal.copy(),
                            action=action.copy(),
                            reward=low_reward,
                            next_observation=next_obs_dict["observation"].copy(),
                            next_achieved_goal=next_obs_dict["achieved_goal"].copy(),
                            done=float(done),
                            info=dict(info or {}),
                        )
                    )
                else:
                    replay.add(low_obs, action, low_reward, low_next_obs, float(done))

                update_many(agent, replay, args, device, critic_losses, actor_losses, q_means)

                if high_replay is not None and high_replay.size >= args.hiro_high_batch_size:
                    for _ in range(args.hiro_high_updates):
                        hs = high_agent.update(high_replay.sample(args.hiro_high_batch_size, device))
                        high_critic_losses.append(hs.critic_loss)
                        if not np.isnan(hs.actor_loss):
                            high_actor_losses.append(hs.actor_loss)
                        high_q_means.append(hs.q_mean)

                ep_return += env_reward
                ep_true_success = max(ep_true_success, float(info.get("is_success", 0.0)))
                true_distances.append(goal_distance(next_obs_dict["achieved_goal"], next_obs_dict["desired_goal"]))

                if args.use_hiro:
                    sg_dist = goal_distance(next_obs_dict["achieved_goal"], current_subgoal)
                    subgoal_distances.append(sg_dist)
                    ep_subgoal_success = max(ep_subgoal_success, subgoal_success(next_obs_dict["achieved_goal"], current_subgoal))
                    # Progress-style high-level reward. More negative if final goal remains far.
                    high_reward_accum += -goal_distance(next_obs_dict["achieved_goal"], next_obs_dict["desired_goal"])
                else:
                    subgoal_distances.append(np.nan)

                obs_dict = next_obs_dict
                total_steps += 1
                steps_since_high += 1

            if high_replay is not None and have_pending_high_segment:
                high_replay.add(high_start_obs, current_delta, high_reward_accum, make_high_level_obs(obs_dict), 1.0)

            if args.use_her:
                her_added = add_episode_with_her(
                    replay,
                    episode_transitions,
                    reward_fn=reward_fn,
                    her_k=args.her_k,
                    use_her=True,
                    future_offset=args.her_future_offset,
                )
                update_many(agent, replay, args, device, critic_losses, actor_losses, q_means)

            should_eval = (episode % args.eval_every == 0) or (episode == 1 and args.eval_on_first_episode)
            if should_eval:
                last_eval = evaluate(agent, args, args.seed, args.eval_episodes, high_agent=high_agent)
                recent_eval_success.append(last_eval[1])
                eval_return_now, eval_success_now, eval_subgoal_success_now, eval_true_dist_now, eval_subgoal_dist_now = last_eval
                if eval_success_now >= best_eval_success:
                    best_eval_success = eval_success_now
                    save_training_checkpoint(
                        checkpoint_dir=args.checkpoint_dir,
                        method=method,
                        task=args.task,
                        reward_type=args.reward_type,
                        seed=args.seed,
                        episode=episode,
                        total_steps=total_steps,
                        agent=agent,
                        high_agent=high_agent,
                        args=args,
                        metrics={
                            "eval_return": eval_return_now,
                            "eval_success": eval_success_now,
                            "eval_subgoal_success": eval_subgoal_success_now,
                            "eval_true_goal_distance": eval_true_dist_now,
                            "eval_subgoal_distance": eval_subgoal_dist_now,
                            "recent_eval_success_mean": float(np.mean(recent_eval_success)) if recent_eval_success else np.nan,
                        },
                    )

            eval_values = last_eval if (should_eval or args.carry_eval_forward) else (np.nan, np.nan, np.nan, np.nan, np.nan)
            eval_return, eval_success, eval_subgoal_success, eval_true_dist, eval_subgoal_dist = eval_values

            writer.writerow([
                episode, total_steps, ep_return, ep_true_success, ep_subgoal_success,
                float(np.mean(true_distances)) if true_distances else np.nan,
                float(np.nanmean(subgoal_distances)) if len(subgoal_distances) else np.nan,
                eval_return, eval_success, eval_subgoal_success, eval_true_dist, eval_subgoal_dist,
                float(np.mean(critic_losses)) if critic_losses else np.nan,
                float(np.mean(actor_losses)) if actor_losses else np.nan,
                float(np.mean(q_means)) if q_means else np.nan,
                float(np.mean(high_critic_losses)) if high_critic_losses else np.nan,
                float(np.mean(high_actor_losses)) if high_actor_losses else np.nan,
                float(np.mean(high_q_means)) if high_q_means else np.nan,
                her_added, replay.size, high_replay.size if high_replay is not None else 0,
            ])
            f.flush()

            eval_success_str = f"{eval_success:.3f}" if not np.isnan(eval_success) else "NA"
            print(
                f"Episode {episode:04d} | method={method} | task={args.task} | reward={args.reward_type} | "
                f"return={ep_return:.2f} | true_success={ep_true_success:.1f} | subgoal_success={ep_subgoal_success:.1f} | "
                f"eval_success={eval_success_str} | true_dist={np.mean(true_distances):.3f} | "
                f"her_added={her_added} | replay={replay.size}"
            )

    env.close()
    print(f"Saved log to: {csv_path}")
    plot_from_csv(csv_path, args.plots_dir, smooth=args.plot_smooth)


if __name__ == "__main__":
    main()
