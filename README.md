# Learning Goal-Conditioned Robotic Manipulation under Sparse Rewards with Hindsight Experience Replay

Authors: Elliot Markovich, Daria Goptsii

This project evaluates reinforcement learning methods for goal-conditioned robotic manipulation under sparse rewards in the FetchReach environment. We compare DDPG, TD3, and SAC under dense and sparse reward settings, then evaluate Hindsight Experience Replay (HER), prioritized HER, curriculum learning, and GoalGAN-style goal sampling.

Results show that all baseline methods solve the task under dense rewards, but sparse rewards make exploration significantly harder. Under sparse rewards, DDPG and TD3 fail to learn reliably, while SAC succeeds but requires many more episodes. HER substantially improves sample efficiency, and prioritized HER gives the best performance among the tested methods. Curriculum learning and GoalGAN-style sampling provide limited or negative gains in this simple low-dimensional task.

The full report can be found in this repository as `COMP 579 Project Report.pdf`.

## Usage

Step 0 — Clone repository and install dependencies:

    git clone https://github.com/dgoptsii/RL-Final-project.git
    cd RL-Final-project
    pip install -r requirements.txt

Step 1 — Run baseline methods:

    python baselines/baseline_ddpg.py
    python baselines/baseline_td3.py
    python baselines/baseline_sac.py

Step 2 — Run HER-based methods:

    python her/train_sac_her.py
    python her/train_sac_prioritized_her.py

Step 3 — Run curriculum and goal sampling methods:

    python curriculum/train_sac_her_curriculum.py
    python curriculum/train_sac_goalgan_her.py

Step 4 — Generate comparison plots:

    python plot_all_results.py

## Methods

The project compares the following methods:

- DDPG
- TD3
- SAC
- SAC + HER
- SAC + Prioritized HER
- SAC + HER + Linear Curriculum
- SAC + HER + Adaptive Curriculum
- SAC + HER + GoalGAN-style goal sampling

## Expected Outputs

Training produces logs, plots, model checkpoints, and summary CSV files. Example output locations include:

    results/her_regular/logs/
    results/her_regular/plots/
    results/her_regular/models/
    results/her_regular/results/her_results.csv

## Main Finding

Goal relabeling with HER is the most important factor for solving the sparse-reward FetchReach task. Prioritized HER further improves convergence, while more complex exploration methods such as curriculum learning and GoalGAN-style sampling do not outperform HER in this environment.
