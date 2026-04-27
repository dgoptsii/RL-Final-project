import argparse
import os
import re

import imageio
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import gymnasium_robotics
from PIL import Image, ImageDraw

gym.register_envs(gymnasium_robotics)


def flatten_obs(obs_dict):
    return np.concatenate(
        [
            obs_dict["observation"],
            obs_dict["achieved_goal"],
            obs_dict["desired_goal"],
        ],
        axis=0,
    ).astype(np.float32)


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, act_limit):
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

    def forward(self, obs):
        return self.act_limit * self.net(obs)


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, act_limit):
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

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def act(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.act_limit


def load_actor(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    algorithm = checkpoint.get("algorithm", "").lower()

    if algorithm == "sac":
        actor = GaussianActor(
            checkpoint["obs_dim"],
            checkpoint["act_dim"],
            checkpoint["hidden_dim"],
            checkpoint["act_limit"],
        ).to(device)
    else:
        actor = MLPActor(
            checkpoint["obs_dim"],
            checkpoint["act_dim"],
            checkpoint["hidden_dim"],
            checkpoint["act_limit"],
        ).to(device)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    return actor, checkpoint


def select_action(actor, obs_t):
    if hasattr(actor, "act"):
        return actor.act(obs_t).cpu().numpy()[0]
    return actor(obs_t).cpu().numpy()[0]


def extract_episode(filename):
    match = re.search(r"episode[_\-]?(\d+)", filename)
    if match is None:
        return None
    return int(match.group(1))


def add_text(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    draw.rectangle((5, 5, 620, 38), fill=(255, 255, 255))
    draw.text((12, 12), text, fill=(0, 0, 0))

    return np.array(img)


@torch.no_grad()
def rollout(env, actor, device, episode_num, algorithm_name, max_steps=50):
    frames = []
    total_return = 0.0
    final_success = 0.0
    final_distance = None

    obs_dict, _ = env.reset()
    obs = flatten_obs(obs_dict)

    for _ in range(max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = select_action(actor, obs_t)

        obs_dict, reward, done, truncated, info = env.step(action)
        obs = flatten_obs(obs_dict)

        total_return += reward
        final_success = float(info.get("is_success", final_success))
        final_distance = np.linalg.norm(
            obs_dict["achieved_goal"] - obs_dict["desired_goal"]
        )

        label = (
            f"{algorithm_name.upper()} | checkpoint ep {episode_num} | "
            f"success={final_success:.0f} | return={total_return:.1f} | "
            f"dist={final_distance:.3f}"
        )

        frame = env.render()
        frames.append(add_text(frame, label))

        if done or truncated:
            break

    return frames, {
        "episode": episode_num,
        "success": final_success,
        "return": total_return,
        "distance": final_distance,
        "steps": len(frames),
    }


def collect_checkpoints(checkpoint_dir):
    checkpoint_files = []

    for f in os.listdir(checkpoint_dir):
        if f.endswith(".pt") and "episode" in f:
            ep = extract_episode(f)
            if ep is not None:
                checkpoint_files.append((ep, f))

    checkpoint_files = sorted(checkpoint_files, key=lambda x: x[0])

    if len(checkpoint_files) == 0:
        raise ValueError(f"No episode checkpoints found in {checkpoint_dir}")

    return checkpoint_files


def select_even_checkpoints(checkpoint_files, max_checkpoints):
    idxs = np.linspace(
        0,
        len(checkpoint_files) - 1,
        min(max_checkpoints, len(checkpoint_files)),
    ).astype(int)

    idxs = sorted(set(idxs))
    return [checkpoint_files[i] for i in idxs]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="FetchReach")
    parser.add_argument("--reward-type", type=str, default="sparse")
    parser.add_argument("--out", type=str, default="results/animations/baseline_learning.gif")
    parser.add_argument("--fps", type=int, default=20)

    parser.add_argument("--max-checkpoints", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--success-streak-stop", type=int, default=3)
    parser.add_argument("--pause-frames", type=int, default=10)

    args = parser.parse_args()
    device = torch.device("cpu")

    checkpoint_files = collect_checkpoints(args.checkpoint_dir)
    selected = select_even_checkpoints(checkpoint_files, args.max_checkpoints)

    print("\nUsing checkpoints in numerical order:")
    for ep, f in selected:
        print(f"episode {ep}: {f}")

    env = gym.make(
        f"{args.task}-v4",
        reward_type=args.reward_type,
        render_mode="rgb_array",
    )

    all_frames = []
    success_streak = 0
    algorithm_name = "baseline"

    for episode_num, file in selected:
        path = os.path.join(args.checkpoint_dir, file)
        actor, checkpoint = load_actor(path, device)
        algorithm_name = checkpoint.get("algorithm", algorithm_name)

        frames, result = rollout(
            env=env,
            actor=actor,
            device=device,
            episode_num=episode_num,
            algorithm_name=algorithm_name,
            max_steps=args.max_steps,
        )

        all_frames.extend(frames)

        for _ in range(args.pause_frames):
            all_frames.append(frames[-1])

        print(
            f"Episode {episode_num:>5} | algo={algorithm_name} | "
            f"success={result['success']:.0f} | "
            f"return={result['return']:.2f} | "
            f"distance={result['distance']:.4f} | "
            f"steps={result['steps']}"
        )

        if result["success"] == 1.0:
            success_streak += 1
        else:
            success_streak = 0

        if success_streak >= args.success_streak_stop:
            print(
                f"\nStopping animation after "
                f"{success_streak} consecutive successful checkpoints."
            )
            break

    env.close()

    if len(all_frames) == 0:
        raise RuntimeError("No frames were generated.")

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    imageio.mimsave(args.out, all_frames, fps=args.fps)
    print(f"\nSaved GIF to: {args.out}")


if __name__ == "__main__":
    main()