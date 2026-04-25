
import argparse
import csv
import json
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

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_all_results import find_x_column, outdir_for_csv, plot_single_run

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

class DDPGAgent:
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
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit

        self.actor = MLPActor(obs_dim, act_dim, hidden_dim, act_limit).to(device)
        self.actor_target = MLPActor(obs_dim, act_dim, hidden_dim, act_limit).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = MLPCritic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target = MLPCritic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act = self.actor(obs_t).cpu().numpy()[0]
        if noise_std > 0.0:
            act = act + noise_std * np.random.randn(*act.shape)
        return np.clip(act, -self.act_limit, self.act_limit)

    def update(self, batch) -> TrainStats:
        obs = batch["obs"]
        acts = batch["acts"]
        rews = batch["rews"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_actions)
            y = rews + self.gamma * (1.0 - done) * target_q

        q = self.critic(obs, acts)
        critic_loss = F.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_actions = self.actor(obs)
        actor_loss = -self.critic(obs, actor_actions).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return TrainStats(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            q_mean=float(q.mean().item()),
        )

    def soft_update(self, net: nn.Module, target_net: nn.Module) -> None:
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

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


def summarize_run(csv_path: str, algorithm: str, task: str, reward_type: str, seed: int, args,
                  model_path: str, last_n: int = 5) -> dict:
    df = pd.read_csv(csv_path)
    eval_success = df["eval_success"].dropna() if "eval_success" in df.columns else pd.Series(dtype=float)
    eval_return = df["eval_return"].dropna() if "eval_return" in df.columns else pd.Series(dtype=float)

    return {
        "algorithm": algorithm,
        "task": task,
        "reward_type": reward_type,
        "seed": seed,
        "episodes": args.episodes,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "final_eval_success": float(eval_success.iloc[-1]) if len(eval_success) else float("nan"),
        "best_eval_success": float(eval_success.max()) if len(eval_success) else float("nan"),
        "last5_eval_success_mean": float(eval_success.tail(last_n).mean()) if len(eval_success) else float("nan"),
        "last5_eval_success_std": float(eval_success.tail(last_n).std(ddof=0)) if len(eval_success) else float("nan"),
        "final_eval_return": float(eval_return.iloc[-1]) if len(eval_return) else float("nan"),
        "best_eval_return": float(eval_return.max()) if len(eval_return) else float("nan"),
        "last5_eval_return_mean": float(eval_return.tail(last_n).mean()) if len(eval_return) else float("nan"),
        "csv_path": csv_path,
        "model_path": model_path,
    }


def write_summary_files(summary: dict, run_summary_dir: str, global_results_dir: str) -> None:
    run_csv = os.path.join(run_summary_dir, "summary.csv")
    run_json = os.path.join(run_summary_dir, "summary.json")
    pd.DataFrame([summary]).to_csv(run_csv, index=False)
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    global_csv = os.path.join(global_results_dir, "baseline_results.csv")
    new_row = pd.DataFrame([summary])
    if os.path.exists(global_csv):
        old = pd.read_csv(global_csv)
        key = ((old["algorithm"] == summary["algorithm"]) &
               (old["task"] == summary["task"]) &
               (old["reward_type"] == summary["reward_type"]) &
               (old["seed"] == summary["seed"]))
        old = old.loc[~key]
        out = pd.concat([old, new_row], ignore_index=True)
    else:
        out = new_row
    out.sort_values(["task", "reward_type", "algorithm", "seed"]).to_csv(global_csv, index=False)


def save_checkpoint(agent, model_path: str, algorithm: str, args, obs_dim: int, act_dim: int, act_limit: float) -> None:
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
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--action-noise", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-on-first-episode", action="store_true")
    parser.add_argument("--carry-eval-forward", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--plot-smooth", type=int, default=10)
    parser.add_argument("--baseline-dir", type=str, default="baselines")
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

    agent = DDPGAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
    )
    replay = ReplayBuffer(obs_dim, act_dim, args.replay_size)

    run_dirs = create_run_dirs(args.baseline_dir, "ddpg", args.task, args.reward_type, args.seed)
    run_name = f"ddpg_{args.task}_{args.reward_type}_seed{args.seed}"
    csv_path = os.path.join(run_dirs["logs"], f"{run_name}.csv")
    model_path = os.path.join(run_dirs["models"], f"{run_name}_final.pt")

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
                    act = agent.select_action(obs, noise_std=args.action_noise)

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
                f"Episode {episode:03d} | algo=ddpg | task={args.task} | reward={args.reward_type} | "
                f"return={ep_return:.2f} | curr_success={ep_curriculum_success:.1f} (recent={recent_curr_str}) | "
                f"true_success={ep_true_success:.1f} (recent={recent_true_str}) | "
                f"eval_success={eval_success_str} | critic_loss={mean_critic_loss:.6f} | "
                f"actor_loss={mean_actor_loss:.6f} | q_mean={mean_q:.6f}"
            )

    env.close()
    save_checkpoint(agent, model_path, "ddpg", args, obs_dim, act_dim, act_limit)
    summary = summarize_run(csv_path, "ddpg", args.task, args.reward_type, args.seed, args, model_path)
    write_summary_files(summary, run_dirs["summaries"], run_dirs["results"])

    print(f"Saved log to: {csv_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved run summary to: {os.path.join(run_dirs['summaries'], 'summary.csv')}")
    print(f"Updated global table: {os.path.join(run_dirs['results'], 'baseline_results.csv')}")
    write_and_plot(csv_path, smooth=args.plot_smooth, plots_dir=run_dirs["plots"])


if __name__ == "__main__":
    main()
