"""
Simplified HIRO-style hierarchy for Fetch environments.

Supports:
1. heuristic high-level subgoals: interpolate from achieved_goal to desired_goal;
2. learned high-level TD3 subgoal policy.

The heuristic mode is recommended first because it isolates HER/SAC behavior and
avoids instability from simultaneously learning two policies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FETCH_SUCCESS_THRESHOLD = 0.05


def goal_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def subgoal_success(achieved_goal: np.ndarray, subgoal: np.ndarray, threshold: float = FETCH_SUCCESS_THRESHOLD) -> float:
    return float(goal_distance(achieved_goal, subgoal) < threshold)


def true_goal_success(achieved_goal: np.ndarray, desired_goal: np.ndarray, threshold: float = FETCH_SUCCESS_THRESHOLD) -> float:
    return float(goal_distance(achieved_goal, desired_goal) < threshold)


def heuristic_subgoal(achieved_goal: np.ndarray, desired_goal: np.ndarray, ratio: float, noise_std: float = 0.0) -> np.ndarray:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    sg = achieved_goal + ratio * (desired_goal - achieved_goal)
    if noise_std > 0.0:
        sg = sg + np.random.normal(0.0, noise_std, size=sg.shape)
    return sg.astype(np.float32)


class HighLevelReplayBuffer:
    def __init__(self, obs_dim: int, subgoal_dim: int, size: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.subgoals = np.zeros((size, subgoal_dim), dtype=np.float32)
        self.rews = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def add(self, obs, subgoal, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.subgoals[self.ptr] = subgoal
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32, device=device),
            "acts": torch.tensor(self.subgoals[idx], dtype=torch.float32, device=device),
            "rews": torch.tensor(self.rews[idx], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "done": torch.tensor(self.done[idx], dtype=torch.float32, device=device),
        }


class HighLevelActor(nn.Module):
    def __init__(self, obs_dim: int, subgoal_dim: int, hidden_dim: int, subgoal_limit: float):
        super().__init__()
        self.subgoal_limit = subgoal_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, subgoal_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.subgoal_limit * self.net(obs)


class HighLevelCritic(nn.Module):
    def __init__(self, obs_dim: int, subgoal_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + subgoal_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, subgoal], dim=-1))


@dataclass
class HighLevelStats:
    critic_loss: float
    actor_loss: float
    q_mean: float


class HighLevelTD3Agent:
    def __init__(
        self,
        obs_dim: int,
        subgoal_dim: int,
        hidden_dim: int,
        subgoal_limit: float,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        device: torch.device,
        policy_noise: float = 0.1,
        noise_clip: float = 0.2,
        policy_delay: int = 2,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.subgoal_limit = subgoal_limit
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.update_count = 0

        self.actor = HighLevelActor(obs_dim, subgoal_dim, hidden_dim, subgoal_limit).to(device)
        self.actor_target = HighLevelActor(obs_dim, subgoal_dim, hidden_dim, subgoal_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = HighLevelCritic(obs_dim, subgoal_dim, hidden_dim).to(device)
        self.critic2 = HighLevelCritic(obs_dim, subgoal_dim, hidden_dim).to(device)
        self.critic1_target = HighLevelCritic(obs_dim, subgoal_dim, hidden_dim).to(device)
        self.critic2_target = HighLevelCritic(obs_dim, subgoal_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

    @torch.no_grad()
    def select_subgoal_delta(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        delta = self.actor(obs_t).cpu().numpy()[0]
        if noise_std > 0.0:
            delta = delta + noise_std * np.random.randn(*delta.shape)
        return np.clip(delta, -self.subgoal_limit, self.subgoal_limit).astype(np.float32)

    def update(self, batch) -> HighLevelStats:
        obs = batch["obs"]
        acts = batch["acts"]
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            noise = torch.randn_like(acts) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_acts = self.actor_target(next_obs) + noise
            next_acts = torch.clamp(next_acts, -self.subgoal_limit, self.subgoal_limit)
            target_q1 = self.critic1_target(next_obs, next_acts)
            target_q2 = self.critic2_target(next_obs, next_acts)
            y = rews + self.gamma * (1.0 - done) * torch.min(target_q1, target_q2)

        q1 = self.critic1(obs, acts)
        q2 = self.critic2(obs, acts)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        self.update_count += 1
        actor_loss_value = np.nan
        if self.update_count % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.soft_update(self.actor, self.actor_target)
            actor_loss_value = float(actor_loss.item())

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        return HighLevelStats(float(critic_loss.item()), float(actor_loss_value), float(q1.mean().item()))

    def soft_update(self, net: nn.Module, target_net: nn.Module) -> None:
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
