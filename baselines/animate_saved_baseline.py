import argparse
import os

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn

gym.register_envs(gymnasium_robotics)


def flatten_obs(obs_dict):
    return np.concatenate(
        [obs_dict["observation"], obs_dict["achieved_goal"], obs_dict["desired_goal"]],
        axis=0,
    ).astype(np.float32)


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
        return torch.tanh(mean) * self.act_limit


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Render an animation from a saved baseline checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to *_final.pt saved by a baseline script.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out", type=str, default=None, help="Output .gif path. Defaults next to checkpoint.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    algorithm = ckpt["algorithm"]
    task = ckpt["task"]
    reward_type = ckpt["reward_type"]
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    act_limit = float(ckpt["act_limit"])
    hidden_dim = int(ckpt["hidden_dim"])

    if algorithm == "sac":
        actor = GaussianActor(obs_dim, act_dim, hidden_dim, act_limit)
    else:
        actor = MLPActor(obs_dim, act_dim, hidden_dim, act_limit)

    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.to(args.device)
    actor.eval()

    env = gym.make(f"{task}-v4", reward_type=reward_type, render_mode="rgb_array")
    frames = []

    for ep in range(args.episodes):
        obs_dict, _ = env.reset(seed=int(ckpt.get("seed", 0)) + 50_000 + ep)
        done = False
        truncated = False

        while not (done or truncated):
            frame = env.render()
            frames.append(frame)

            obs = flatten_obs(obs_dict)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            action = actor(obs_t).cpu().numpy()[0]
            obs_dict, _, done, truncated, _ = env.step(action)

    env.close()

    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out = os.path.join(os.path.dirname(args.checkpoint), f"{stem}.gif")

    imageio.mimsave(out, frames, fps=args.fps)
    print(f"Saved animation to: {out}")


if __name__ == "__main__":
    main()
