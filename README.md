# Learning Goal-Conditioned Robotic Manipulation under Sparse Rewards with HER

This repository contains the final project for **COMP 579: Reinforcement Learning** at McGill University. Full report can be found in the root folder of the repository. 

The project studies goal-conditioned robotic manipulation under sparse rewards using the `FetchReach` environment. We compare standard off-policy reinforcement learning methods and evaluate whether Hindsight Experience Replay (HER), prioritized HER, curriculum learning, and GoalGAN-style goal sampling improve learning efficiency.

## Project Overview

Sparse rewards make robotic reinforcement learning difficult because the agent receives useful feedback only when it successfully reaches the goal. In the `FetchReach` task, a robotic arm must move its end-effector to a target position in 3D space.

We evaluate:

- DDPG
- TD3
- SAC
- SAC + HER
- SAC + Prioritized HER
- SAC + HER + Linear Curriculum
- SAC + HER + Adaptive Curriculum
- SAC + HER + GoalGAN-style goal sampling


## Key Results

| Method | Reward Type | Final Success | Episodes to 90% Success |
|---|---:|---:|---:|
| DDPG | Dense | 1.00 | 180 |
| TD3 | Dense | 1.00 | 220 |
| SAC | Dense | 1.00 | 90 |
| DDPG | Sparse | 0.00 | — |
| TD3 | Sparse | 0.10 | — |
| SAC | Sparse | 1.00 | 350 |
| SAC + HER | Sparse | 1.00 | 190 |
| SAC + Prioritized HER | Sparse | 1.00 | 150 |
| SAC + HER + Linear Curriculum | Sparse | 1.00 | 180 |
| SAC + HER + Adaptive Curriculum | Sparse | 1.00 | 200 |
| SAC + HER + GoalGAN | Sparse | 1.00 | 210 |

## Repository Structure

```text
RL-Final-project/
├── baselines/              # DDPG, TD3, and SAC baseline implementations
├── her/                    # HER and prioritized HER implementations
├── curriculum/             # Curriculum learning and GoalGAN-style methods
├── animations/             # Saved environment rollouts / visualizations
├── plot_all_results.py     # Script for plotting result comparisons
├── requirements.txt        # Python dependencies
└── README.md
