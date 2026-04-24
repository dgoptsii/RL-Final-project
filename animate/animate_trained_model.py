"""
Animate a trained SAC / SAC+HER / SAC+HIRO / SAC+HIRO+HER Fetch agent.

This script loads a checkpoint saved with checkpoint_utils.py and exports a GIF
or MP4 from Gymnasium-Robotics Fetch environments.

Example:

    python animate_trained_model.py \
        --checkpoint checkpoints_hiro_her/sac_her_hiro_FetchReach_dense_seed0_best.pt \
        --task FetchReach \
        --reward-type dense \
        --use-hiro \
        --hiro-high-level-mode heuristic \
        --episodes 3 \
        --output videos/fetchreach_hiro_her.gif

Notes:
- For heuristic HIRO, no high-level network is needed.
- For learned HIRO, the checkpoint must contain the high-level agent.
- GIF writing requires imageio. MP4 writing may require imageio-ffmpeg.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio
import numpy as np
import torch

from checkpoints.checkpoint_utils import load_checkpoint_payload, load_high_agent_state, load_sac_agent_state
from hiro import HighLevelTD3Agent, goal_distance, heuristic_subgoal, subgoal_success
from train_sac_hiro_her import SACAgent, flatten_obs_dict, make_low_level_obs

gym.register_envs(gymnasium_robotics)


def make_env(task: str, reward_type: str, seed: int, render_mode: str = "rgb_array"):
    env_name = f"{task}-v4"
    env = gym.make(env_name, reward_type=reward_type, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def build_args_from_checkpoint(payload: Dict, cli_args: argparse.Namespace) -> SimpleNamespace:
    saved = payload.get("args", {}) or {}

    def get(name: str, default):
        return getattr(cli_args, name, None) if getattr(cli_args, name, None) is not None else saved.get(name, default)

    # Only values needed for model reconstruction and HIRO subgoal generation.
    return SimpleNamespace(
        task=cli_args.task or payload.get("task") or saved.get("task", "FetchReach"),
        reward_type=cli_args.reward_type or payload.get("reward_type") or saved.get("reward_type", "dense"),
        device=cli_args.device,
        hidden_dim=int(get("hidden_dim", 256)),
        actor_lr=float(get("actor_lr", 3e-4)),
        critic_lr=float(get("critic_lr", 3e-4)),
        gamma=float(get("gamma", 0.98)),
        tau=float(get("tau", 0.005)),
        alpha=float(get("alpha", 0.2)),
        use_hiro=bool(cli_args.use_hiro or saved.get("use_hiro", False)),
        hiro_interval=int(get("hiro_interval", 10)),
        hiro_high_level_mode=cli_args.hiro_high_level_mode or saved.get("hiro_high_level_mode", "heuristic"),
        hiro_subgoal_ratio=float(get("hiro_subgoal_ratio", 0.5)),
        hiro_subgoal_noise=float(cli_args.hiro_subgoal_noise if cli_args.hiro_subgoal_noise is not None else 0.0),
        hiro_subgoal_limit=float(get("hiro_subgoal_limit", 0.25)),
        hiro_high_hidden_dim=int(get("hiro_high_hidden_dim", 256)),
        hiro_high_actor_lr=float(get("hiro_high_actor_lr", 1e-4)),
        hiro_high_critic_lr=float(get("hiro_high_critic_lr", 3e-4)),
        hiro_high_gamma=float(get("hiro_high_gamma", 0.98)),
        hiro_high_tau=float(get("hiro_high_tau", 0.005)),
        hiro_high_noise=float(cli_args.hiro_high_noise if cli_args.hiro_high_noise is not None else 0.0),
    )


def select_hiro_subgoal(args: SimpleNamespace, obs_dict: Dict, high_agent: Optional[HighLevelTD3Agent]) -> Tuple[np.ndarray, np.ndarray]:
    achieved = obs_dict["achieved_goal"].copy()
    desired = obs_dict["desired_goal"].copy()
    if args.hiro_high_level_mode == "heuristic":
        subgoal = heuristic_subgoal(
            achieved,
            desired,
            ratio=args.hiro_subgoal_ratio,
            noise_std=args.hiro_subgoal_noise,
        )
        delta = subgoal - achieved
        return subgoal.astype(np.float32), delta.astype(np.float32)

    if high_agent is None:
        raise ValueError("Learned HIRO requested, but no high_agent was constructed.")
    high_obs = flatten_obs_dict(obs_dict)
    delta = high_agent.select_subgoal_delta(high_obs, noise_std=args.hiro_high_noise)
    subgoal = achieved + delta
    return subgoal.astype(np.float32), delta.astype(np.float32)


def reconstruct_agents(payload: Dict, args: SimpleNamespace, env) -> Tuple[SACAgent, Optional[HighLevelTD3Agent]]:
    device = torch.device(args.device)
    obs_dict, _ = env.reset(seed=0)
    low_obs = make_low_level_obs(obs_dict, obs_dict["desired_goal"])
    low_obs_dim = low_obs.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = SACAgent(
        low_obs_dim,
        act_dim,
        act_limit,
        args.hidden_dim,
        args.actor_lr,
        args.critic_lr,
        args.gamma,
        args.tau,
        device,
        args.alpha,
    )
    load_sac_agent_state(agent, payload)

    high_agent = None
    if args.use_hiro and args.hiro_high_level_mode == "learned":
        high_obs_dim = flatten_obs_dict(obs_dict).shape[0]
        subgoal_dim = obs_dict["achieved_goal"].shape[0]
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
        load_high_agent_state(high_agent, payload)

    agent.actor.eval()
    if high_agent is not None:
        high_agent.actor.eval()
    return agent, high_agent


def rollout_and_collect_frames(agent: SACAgent, high_agent: Optional[HighLevelTD3Agent], args: SimpleNamespace, seed: int, episode_idx: int):
    env = make_env(args.task, args.reward_type, seed + episode_idx, render_mode="rgb_array")
    obs_dict, _ = env.reset(seed=seed + episode_idx)
    frames = []
    done = False
    truncated = False
    ep_return = 0.0
    ep_success = 0.0
    ep_subgoal_success = 0.0
    steps_since_high = args.hiro_interval
    subgoal = obs_dict["desired_goal"].copy()

    frame = env.render()
    if frame is not None:
        frames.append(frame)

    while not (done or truncated):
        if args.use_hiro and steps_since_high >= args.hiro_interval:
            subgoal, _ = select_hiro_subgoal(args, obs_dict, high_agent)
            steps_since_high = 0

        goal = subgoal if args.use_hiro else obs_dict["desired_goal"]
        low_obs = make_low_level_obs(obs_dict, goal)
        action = agent.select_action(low_obs, stochastic=False)

        next_obs_dict, reward, done, truncated, info = env.step(action)
        ep_return += float(reward)
        ep_success = max(ep_success, float(info.get("is_success", 0.0)))
        if args.use_hiro:
            ep_subgoal_success = max(ep_subgoal_success, subgoal_success(next_obs_dict["achieved_goal"], subgoal))

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_dict = next_obs_dict
        steps_since_high += 1

    final_true_distance = goal_distance(obs_dict["achieved_goal"], obs_dict["desired_goal"])
    final_subgoal_distance = goal_distance(obs_dict["achieved_goal"], subgoal) if args.use_hiro else np.nan
    env.close()

    stats = {
        "episode": episode_idx,
        "return": ep_return,
        "success": ep_success,
        "subgoal_success": ep_subgoal_success if args.use_hiro else np.nan,
        "final_true_distance": final_true_distance,
        "final_subgoal_distance": final_subgoal_distance,
        "num_frames": len(frames),
    }
    return frames, stats


def save_video(frames, output_path: str, fps: int) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    suffix = Path(output_path).suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(output_path, frames, fps=fps)
    elif suffix in {".mp4", ".m4v"}:
        imageio.mimsave(output_path, frames, fps=fps, macro_block_size=16)
    else:
        raise ValueError("Output must end with .gif or .mp4")
    print(f"Saved animation to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a trained SAC / SAC+HER / SAC+HIRO / SAC+HIRO+HER Fetch model.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint saved by checkpoint_utils.py")
    parser.add_argument("--task", default=None, choices=["FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace"])
    parser.add_argument("--reward-type", dest="reward_type", default=None, choices=["dense", "sparse"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default="videos/fetch_policy.gif")
    parser.add_argument("--use-hiro", action="store_true", help="Use HIRO subgoals during animation")
    parser.add_argument("--hiro-high-level-mode", choices=["heuristic", "learned"], default=None)
    parser.add_argument("--hiro-subgoal-noise", type=float, default=0.0, help="Usually keep 0 for clean evaluation videos")
    parser.add_argument("--hiro-high-noise", type=float, default=0.0, help="Usually keep 0 for clean evaluation videos")
    args_cli = parser.parse_args()

    payload = load_checkpoint_payload(args_cli.checkpoint, device=args_cli.device)
    args = build_args_from_checkpoint(payload, args_cli)

    env = make_env(args.task, args.reward_type, args_cli.seed, render_mode="rgb_array")
    agent, high_agent = reconstruct_agents(payload, args, env)
    env.close()

    all_frames = []
    all_stats = []
    for ep in range(args_cli.episodes):
        frames, stats = rollout_and_collect_frames(agent, high_agent, args, args_cli.seed, ep)
        all_frames.extend(frames)
        all_stats.append(stats)
        print(
            f"Episode {ep}: return={stats['return']:.3f}, success={stats['success']:.1f}, "
            f"subgoal_success={stats['subgoal_success']}, true_dist={stats['final_true_distance']:.4f}, "
            f"subgoal_dist={stats['final_subgoal_distance']}, frames={stats['num_frames']}"
        )

    if not all_frames:
        raise RuntimeError("No frames were rendered. Check that the environment supports render_mode='rgb_array'.")
    save_video(all_frames, args_cli.output, args_cli.fps)


if __name__ == "__main__":
    main()
