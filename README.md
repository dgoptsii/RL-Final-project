# RL Fetch Baselines

## Overview
This project implements baseline reinforcement learning algorithms for Fetch robotics tasks:
- DDPG
- TD3
- SAC

Each script trains an agent, logs results, and generates plots automatically.

---

## Files

- baseline_ddpg.py → DDPG baseline
- baseline_td3.py → TD3 baseline
- baseline_sac.py → SAC baseline
- plot_results.py → Plot training results
- train_baselines_optuna.py → Hyperparameter tuning (Optuna)
- requirements_fetch_project.txt → Dependencies

---

## Setup

pip install -r requirements_fetch_project.txt

---

## Run

Example:

python baseline_td3.py --task FetchReach --reward-type dense --episodes 300

---

## Outputs

- Logs → logs/
- Plots → plots/

---

## Plot manually

python plot_results.py --csv logs/your_run.csv

---

## Hyperparameter tuning

python train_baselines_optuna.py --algorithm td3 --task FetchReach --n-trials 20

---

## VS Code

launch.json contains ready-to-use configs for running and debugging.
