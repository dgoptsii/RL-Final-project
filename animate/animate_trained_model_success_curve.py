"""
Animate a trained SAC / SAC+HER / SAC+HIRO / SAC+HIRO+HER Fetch agent.

Supports three modes:
  1. all             -> save the first N episodes exactly as rolled out
  2. first_success   -> search episodes, save only the first successful episode
  3. learning_curve  -> search episodes and build a short video that looks like:
                        bad episode -> better episode -> better episode -> success

Important:
- If you only have ONE checkpoint, learning_curve mode does not show the real training
  process over time. It shows selected evaluation episodes from the same trained model,
  ordered from worse to better. To show the true training curve, you need checkpoints
  saved at different training episodes.

Example first successful episode:
  python animate_trained_model_success_curve.py \
    --checkpoint checkpoints_animation/best_model.pt \
    --task FetchPickAndPlace \
    --reward-type sparse \
    --use-hiro \
    --mode first_success \
    --max-search-episodes 100 \
    --fps 10 \
    --output successful_episode.gif

Example learning-curve style animation:
  python animate_trained_model_success_curve.py \
    --checkpoint checkpoints_animation/best_model.pt \
    --task FetchPickAndPlace \
    --reward-type sparse \
    --use-hiro \
    --mode learning_curve \
    --max-search-episodes 100 \
    --target-clips 4 \
    --fps 10 \
    --output learning_curve_style.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio
import numpy as np
import torch

from checkpoints.checkpoint_utils import load_checkpoint_payload, load_high_agent_state, load_sac_agent_state
from hiro import HighLevelTD3Agent, goal_distance, heuristic_subgoal, subgoal_success
from train_sac_hiro_her import SACAgent, flatten_obs_dict, make_low_level_obs

gym.register_envs(gymnasium_robotics)

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


FETCH_TASKS = ["FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace"]


def make_env(task: str, reward_type: str, seed: int, render_mode: str = "rgb_array"):
    env_name = f"{task}-v4"
    env = gym.make(env_name, reward_type=reward_type, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def build_args_from_checkpoint(payload: Dict, cli_args: argparse.Namespace) -> SimpleNamespace:
    saved = payload.get("args", {}) or {}

    def get(name: str, default):
        value = getattr(cli_args, name, None)
        return value if value is not None else saved.get(name, default)

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
        return subgoal.astype(np.float32), (subgoal - achieved).astype(np.float32)

    if high_agent is None:
        raise ValueError("Learned HIRO requested, but high-level agent was not constructed.")

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


def rollout_and_collect_frames(
    agent: SACAgent,
    high_agent: Optional[HighLevelTD3Agent],
    args: SimpleNamespace,
    seed: int,
    episode_idx: int,
    frame_stride: int = 1,
):
    env = make_env(args.task, args.reward_type, seed + episode_idx, render_mode="rgb_array")
    obs_dict, _ = env.reset(seed=seed + episode_idx)

    frames: List[np.ndarray] = []
    done = False
    truncated = False
    ep_return = 0.0
    ep_success = 0.0
    ep_subgoal_success = 0.0
    steps_since_high = args.hiro_interval
    step = 0
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
            ep_subgoal_success = max(
                ep_subgoal_success,
                subgoal_success(next_obs_dict["achieved_goal"], subgoal),
            )

        step += 1
        if step % max(1, frame_stride) == 0:
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


def title_frame_like(frame: np.ndarray, text: str, repeat: int = 12) -> List[np.ndarray]:
    """Create repeated title frames. If PIL is unavailable, return the original frame."""
    if not PIL_AVAILABLE:
        return [frame] * max(1, repeat)

    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    box_h = max(70, h // 6)
    draw.rectangle([0, 0, w, box_h], fill=(0, 0, 0, 155))

    try:
        font = ImageFont.truetype("Arial.ttf", size=max(18, h // 28))
    except Exception:
        font = ImageFont.load_default()

    draw.text((16, 18), text, fill=(255, 255, 255, 255), font=font)
    return [np.array(img)] * max(1, repeat)


def annotate_clip(frames: List[np.ndarray], label: str, title_hold_frames: int) -> List[np.ndarray]:
    if not frames:
        return []
    return title_frame_like(frames[0], label, repeat=title_hold_frames) + frames


def save_video(frames: List[np.ndarray], output_path: str, fps: int) -> None:
    if not frames:
        raise RuntimeError("No frames to save.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    suffix = Path(output_path).suffix.lower()

    if suffix == ".gif":
        imageio.mimsave(output_path, frames, fps=fps)
    elif suffix in {".mp4", ".m4v"}:
        imageio.mimsave(output_path, frames, fps=fps, macro_block_size=16)
    else:
        raise ValueError("Output must end with .gif, .mp4, or .m4v")

    print(f"Saved animation to: {output_path}")


def print_stats(stats: Dict) -> None:
    print(
        f"Episode {stats['episode']}: return={stats['return']:.3f}, "
        f"success={stats['success']:.1f}, "
        f"subgoal_success={stats['subgoal_success']}, "
        f"true_dist={stats['final_true_distance']:.4f}, "
        f"subgoal_dist={stats['final_subgoal_distance']}, "
        f"frames={stats['num_frames']}"
    )


def collect_candidates(agent, high_agent, args, seed: int, max_search_episodes: int, frame_stride: int):
    candidates = []
    for ep in range(max_search_episodes):
        frames, stats = rollout_and_collect_frames(agent, high_agent, args, seed, ep, frame_stride=frame_stride)
        print_stats(stats)
        candidates.append({"frames": frames, "stats": stats})
    return candidates


def select_first_success(candidates: List[Dict]) -> Optional[Dict]:
    for c in candidates:
        if c["stats"]["success"] >= 1.0:
            return c
    return None


def select_learning_curve_clips(candidates: List[Dict], target_clips: int) -> List[Dict]:
    """
    Choose clips ordered from bad -> better -> success.
    Priority:
      - include one clearly bad failed episode if available
      - include failed episodes with improving final_true_distance
      - include successful episode at end if available
    """
    if not candidates:
        return []

    # Lower distance is better. Success is best.
    failed = [c for c in candidates if c["stats"]["success"] < 1.0]
    success = [c for c in candidates if c["stats"]["success"] >= 1.0]

    chosen: List[Dict] = []

    if failed:
        # Worst failure first.
        worst = max(failed, key=lambda c: c["stats"]["final_true_distance"])
        chosen.append(worst)

        # Then closer failed attempts, if enough are available.
        failed_sorted = sorted(failed, key=lambda c: c["stats"]["final_true_distance"], reverse=True)
        needed_failed = max(0, target_clips - 1 - (1 if success else 0))
        if needed_failed > 0:
            indices = np.linspace(0, len(failed_sorted) - 1, num=min(needed_failed + 1, len(failed_sorted)), dtype=int).tolist()
            for idx in indices[1:]:
                candidate = failed_sorted[idx]
                if all(candidate is not c for c in chosen):
                    chosen.append(candidate)
                if len(chosen) >= target_clips - (1 if success else 0):
                    break

    # If no failures, choose successful episodes by distance, from worse success to best success.
    if not chosen and success:
        success_sorted_worst_to_best = sorted(success, key=lambda c: c["stats"]["final_true_distance"], reverse=True)
        chosen.extend(success_sorted_worst_to_best[: max(1, target_clips - 1)])

    if success:
        # Best success at the end.
        best_success = min(success, key=lambda c: c["stats"]["final_true_distance"])
        if any(best_success is c for c in chosen):
            chosen.remove(best_success)
        chosen.append(best_success)

    # If still too few clips, fill with best remaining episodes.
    if len(chosen) < target_clips:
        remaining = sorted(candidates, key=lambda c: (c["stats"]["success"], -c["stats"]["final_true_distance"]))
        for c in remaining:
            if all(c is not x for x in chosen):
                chosen.append(c)
            if len(chosen) >= target_clips:
                break

    return chosen[:target_clips]


def build_labeled_sequence(clips: List[Dict], title_hold_frames: int) -> List[np.ndarray]:
    all_frames: List[np.ndarray] = []
    n = len(clips)

    for i, clip in enumerate(clips):
        s = clip["stats"]
        if s["success"] >= 1.0:
            label = f"Clip {i + 1}/{n}: success | return={s['return']:.1f} | dist={s['final_true_distance']:.3f}"
        elif i == 0:
            label = f"Clip {i + 1}/{n}: early failed attempt | return={s['return']:.1f} | dist={s['final_true_distance']:.3f}"
        else:
            label = f"Clip {i + 1}/{n}: closer attempt | return={s['return']:.1f} | dist={s['final_true_distance']:.3f}"
        all_frames.extend(annotate_clip(clip["frames"], label, title_hold_frames))

    return all_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate SAC / SAC+HER / SAC+HIRO / SAC+HIRO+HER Fetch checkpoints."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint saved by checkpoint_utils.py")
    parser.add_argument("--task", default=None, choices=FETCH_TASKS)
    parser.add_argument("--reward-type", dest="reward_type", default=None, choices=["dense", "sparse"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", default="videos/fetch_policy.gif")
    parser.add_argument("--use-hiro", action="store_true", help="Use HIRO subgoals during animation")
    parser.add_argument("--hiro-high-level-mode", choices=["heuristic", "learned"], default=None)
    parser.add_argument("--hiro-subgoal-noise", type=float, default=0.0, help="Usually keep 0 for clean evaluation videos")
    parser.add_argument("--hiro-high-noise", type=float, default=0.0, help="Usually keep 0 for clean evaluation videos")

    parser.add_argument(
        "--mode",
        choices=["all", "first_success", "learning_curve"],
        default="all",
        help="all: save N episodes; first_success: save only first successful episode; learning_curve: bad -> better -> success montage.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Used only in --mode all")
    parser.add_argument("--max-search-episodes", type=int, default=100, help="Used by first_success and learning_curve")
    parser.add_argument("--target-clips", type=int, default=4, help="Number of clips in learning_curve mode")
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth frame; use 2 or 3 to make smaller GIFs")
    parser.add_argument("--title-hold-frames", type=int, default=12, help="How long each text title card is shown")

    args_cli = parser.parse_args()

    payload = load_checkpoint_payload(args_cli.checkpoint, device=args_cli.device)
    args = build_args_from_checkpoint(payload, args_cli)

    env = make_env(args.task, args.reward_type, args_cli.seed, render_mode="rgb_array")
    agent, high_agent = reconstruct_agents(payload, args, env)
    env.close()

    if args_cli.mode == "all":
        all_frames: List[np.ndarray] = []
        for ep in range(args_cli.episodes):
            frames, stats = rollout_and_collect_frames(agent, high_agent, args, args_cli.seed, ep, frame_stride=args_cli.frame_stride)
            print_stats(stats)
            label = f"Episode {ep}: success={stats['success']:.0f} | return={stats['return']:.1f} | dist={stats['final_true_distance']:.3f}"
            all_frames.extend(annotate_clip(frames, label, args_cli.title_hold_frames))
        save_video(all_frames, args_cli.output, args_cli.fps)
        return

    candidates = collect_candidates(
        agent,
        high_agent,
        args,
        seed=args_cli.seed,
        max_search_episodes=args_cli.max_search_episodes,
        frame_stride=args_cli.frame_stride,
    )

    if args_cli.mode == "first_success":
        clip = select_first_success(candidates)
        if clip is None:
            best = min(candidates, key=lambda c: c["stats"]["final_true_distance"])
            print("No successful episode found. Saving closest attempt instead.")
            clip = best
        frames = build_labeled_sequence([clip], args_cli.title_hold_frames)
        save_video(frames, args_cli.output, args_cli.fps)
        return

    if args_cli.mode == "learning_curve":
        clips = select_learning_curve_clips(candidates, target_clips=args_cli.target_clips)
        print("\nSelected clips for learning-curve-style animation:")
        for c in clips:
            print_stats(c["stats"])
        frames = build_labeled_sequence(clips, args_cli.title_hold_frames)
        save_video(frames, args_cli.output, args_cli.fps)
        return


if __name__ == "__main__":
    main()
