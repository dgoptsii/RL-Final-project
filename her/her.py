"""
Correct HER utilities for Fetch-style Gymnasium-Robotics environments.

Main correction: rewards are recomputed with env.compute_reward(...), not with a
hand-written approximation. This is safer across FetchReach, FetchPush,
FetchSlide, and FetchPickAndPlace because the environment owns the exact reward
logic for its achieved_goal / desired_goal representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np


RewardFn = Callable[[np.ndarray, np.ndarray, Optional[Dict]], float]


def flatten_goal_obs(observation: np.ndarray, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
    return np.concatenate([observation, achieved_goal, desired_goal], axis=0).astype(np.float32)


def env_compute_reward(env, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Optional[Dict] = None) -> float:
    """Return scalar reward using the environment's own compute_reward method."""
    if info is None:
        info = {}
    reward = env.unwrapped.compute_reward(achieved_goal, desired_goal, info)
    # Gymnasium-Robotics may return np scalar / array depending on input shape.
    reward_arr = np.asarray(reward, dtype=np.float32)
    return float(reward_arr.reshape(-1)[0])


@dataclass
class EpisodeTransition:
    observation: np.ndarray
    achieved_goal: np.ndarray
    desired_goal: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    next_achieved_goal: np.ndarray
    done: float
    info: Dict


def make_transition_from_obs_dict(
    obs_dict: Dict,
    action: np.ndarray,
    reward: float,
    next_obs_dict: Dict,
    done: float,
    info: Optional[Dict] = None,
    desired_goal_override: Optional[np.ndarray] = None,
) -> EpisodeTransition:
    goal = desired_goal_override if desired_goal_override is not None else obs_dict["desired_goal"]
    return EpisodeTransition(
        observation=obs_dict["observation"].copy(),
        achieved_goal=obs_dict["achieved_goal"].copy(),
        desired_goal=goal.copy(),
        action=action.copy(),
        reward=float(reward),
        next_observation=next_obs_dict["observation"].copy(),
        next_achieved_goal=next_obs_dict["achieved_goal"].copy(),
        done=float(done),
        info=dict(info or {}),
    )


def add_episode_with_her(
    replay_buffer,
    episode: List[EpisodeTransition],
    reward_fn: RewardFn,
    her_k: int = 4,
    use_her: bool = True,
    future_offset: int = 0,
) -> int:
    """
    Add original transitions plus HER-relabeled transitions using the future strategy.

    Args:
        replay_buffer: replay buffer with .add(obs, act, rew, next_obs, done)
        episode: transitions from one episode
        reward_fn: callable (achieved_goal, desired_goal, info) -> scalar reward.
                   In training, pass lambda ag, dg, info: env_compute_reward(env, ag, dg, info).
        her_k: number of relabeled transitions per original transition
        use_her: if false, only original transitions are added
        future_offset: 0 allows relabeling with current/future next achieved goal;
                       1 restricts to strictly future transitions when available.

    Returns:
        Number of HER transitions added.
    """
    if not episode:
        return 0

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
