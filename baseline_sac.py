
import argparse
import csv
import os
import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from plot_results import find_x_column, outdir_for_csv, plot_single_run

gym.register_envs(gymnasium_robotics)

FETCH_SUCCESS_THRESHOLD = 0.05


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten_obs(obs_dict):
    return np.concatenate(
        [obs_dict["observation"], obs_dict["achieved_goal"], obs_dict["desired_goal"]],
        axis=0,
    ).astype(np.float32)


def check_true_goal_success(achieved_goal: np.ndarray, true_goal: np.ndarray) -> float:
    return float(np.linalg.norm(achieved_goal - true_goal) < FETCH_SUCCESS_THRESHOLD)


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


class MLPActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, act_limit: float):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act_limit * self.net(obs)


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


@dataclass
class TrainStats:
    critic_loss: float
    actor_loss: float
    q_mean: float


def make_env(task: str, reward_type: str, seed: int):
    env_name = f"{task}-v4"
    env = gym.make(env_name, reward_type=reward_type)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


@torch.no_grad()
def evaluate(agent, task: str, reward_type: str, seed: int, episodes: int) -> tuple[float, float]:
    env = make_env(task, reward_type, seed + 10_000)
    returns = []
    successes = []

    for ep in range(episodes):
        obs_dict, _ = env.reset(seed=seed + 10_000 + ep)
        obs = flatten_obs(obs_dict)
        done = False
        truncated = False
        ep_return = 0.0
        ep_success = 0.0

        while not (done or truncated):
            act = agent.select_action(obs, noise_std=0.0)
            next_obs_dict, rew, done, truncated, info = env.step(act)
            obs = flatten_obs(next_obs_dict)
            ep_return += rew
            ep_success = max(ep_success, float(info.get("is_success", 0.0)))

        returns.append(ep_return)
        successes.append(ep_success)

    env.close()
    return float(np.mean(returns)), float(np.mean(successes))


def write_and_plot(csv_path: str, smooth: int = 10, plots_dir: str = "plots") -> None:
    df = pd.read_csv(csv_path)
    df["__index__"] = range(len(df))
    x_col = find_x_column(df)
    run_outdir = outdir_for_csv(plots_dir, csv_path)
    plot_single_run(df, x_col, run_outdir, smooth)
    print(f"Plots saved to: {run_outdir}/")

LOG_STD_MIN = -20
LOG_STD_MAX = 2


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
        std = torch.exp(log_std)
        return mean, std

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
    ):
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
    def select_action(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if noise_std > 0.0:
            action, _, _ = self.actor.sample(obs_t)
        else:
            _, _, action = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

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
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
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
        q1_new = self.critic1(obs, new_actions)
        q2_new = self.critic2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return TrainStats(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            q_mean=float(q1.mean().item()),
        )

    def soft_update(self, net: nn.Module, target_net: nn.Module) -> None:
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="FetchReach",
                        choices=["FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace"])
    parser.add_argument("--reward-type", type=str, default="dense", choices=["dense", "sparse"])
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
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-on-first-episode", action="store_true")
    parser.add_argument("--carry-eval-forward", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--plot-smooth", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

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
    )
    replay = ReplayBuffer(obs_dim, act_dim, args.replay_size)

    csv_path = os.path.join(args.log_dir, f"sac_{args.task}_{args.reward_type}_seed{args.seed}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "train_return", "train_curriculum_success", "train_true_success",
            "eval_return", "eval_success", "critic_loss", "actor_loss", "q_mean"
        ])

        total_steps = 0
        recent_curriculum_success = deque(maxlen=10)
        recent_true_success = deque(maxlen=10)
        last_eval_return = np.nan
        last_eval_success = np.nan

        for episode in range(1, args.episodes + 1):
            obs_dict, _ = env.reset()
            true_goal = obs_dict["desired_goal"].copy()
            obs = flatten_obs(obs_dict)
            done = False
            truncated = False
            ep_return = 0.0
            ep_curriculum_success = 0.0
            ep_true_success = 0.0
            critic_losses, actor_losses, q_means = [], [], []

            while not (done or truncated):
                if total_steps < args.start_steps:
                    act = env.action_space.sample()
                else:
                    act = agent.select_action(obs, noise_std=1.0)

                next_obs_dict, rew, done, truncated, info = env.step(act)
                next_obs = flatten_obs(next_obs_dict)
                replay.add(obs, act, rew, next_obs, float(done))
                obs = next_obs
                ep_return += rew

                ep_curriculum_success = max(ep_curriculum_success, float(info.get("is_success", 0.0)))
                achieved = next_obs_dict["achieved_goal"]
                ep_true_success = max(ep_true_success, check_true_goal_success(achieved, true_goal))

                total_steps += 1

                if replay.size >= args.batch_size:
                    for _ in range(args.updates_per_step):
                        batch = replay.sample(args.batch_size, device)
                        stats = agent.update(batch)
                        critic_losses.append(stats.critic_loss)
                        actor_losses.append(stats.actor_loss)
                        q_means.append(stats.q_mean)

            recent_curriculum_success.append(ep_curriculum_success)
            recent_true_success.append(ep_true_success)

            should_eval = (episode % args.eval_every == 0) or (episode == 1 and args.eval_on_first_episode)
            if should_eval:
                last_eval_return, last_eval_success = evaluate(
                    agent, task=args.task, reward_type=args.reward_type, seed=args.seed, episodes=args.eval_episodes
                )

            if args.carry_eval_forward:
                eval_return, eval_success = last_eval_return, last_eval_success
            else:
                eval_return = last_eval_return if should_eval else np.nan
                eval_success = last_eval_success if should_eval else np.nan

            mean_critic_loss = float(np.mean(critic_losses)) if critic_losses else np.nan
            mean_actor_loss = float(np.mean(actor_losses)) if actor_losses else np.nan
            mean_q = float(np.mean(q_means)) if q_means else np.nan

            writer.writerow([
                episode, ep_return, ep_curriculum_success, ep_true_success,
                eval_return, eval_success, mean_critic_loss, mean_actor_loss, mean_q,
            ])
            f.flush()

            eval_success_str = f"{eval_success:.3f}" if not np.isnan(eval_success) else "NA"
            recent_curr = float(np.mean(recent_curriculum_success)) if recent_curriculum_success else np.nan
            recent_true = float(np.mean(recent_true_success)) if recent_true_success else np.nan
            recent_curr_str = f"{recent_curr:.3f}" if not np.isnan(recent_curr) else "NA"
            recent_true_str = f"{recent_true:.3f}" if not np.isnan(recent_true) else "NA"

            print(
                f"Episode {episode:03d} | algo=sac | task={args.task} | reward={args.reward_type} | "
                f"return={ep_return:.2f} | curr_success={ep_curriculum_success:.1f} (recent={recent_curr_str}) | "
                f"true_success={ep_true_success:.1f} (recent={recent_true_str}) | "
                f"eval_success={eval_success_str} | critic_loss={mean_critic_loss:.6f} | "
                f"actor_loss={mean_actor_loss:.6f} | q_mean={mean_q:.6f}"
            )

    env.close()
    print(f"Saved log to: {csv_path}")
    write_and_plot(csv_path, smooth=args.plot_smooth, plots_dir=args.plots_dir)


if __name__ == "__main__":
    main()
