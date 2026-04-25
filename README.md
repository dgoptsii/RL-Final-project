

## Recommended run order

Run these first on FetchReach sparse:

1. SAC baseline - FetchReach sparse
2. SAC + HER - FetchReach sparse
3. SAC + prioritized HER - FetchReach sparse
4. SAC + HER + linear curriculum - FetchReach sparse
5. SAC + HER + adaptive curriculum - FetchReach sparse
6. SAC + HER + GoalGAN sampler - FetchReach sparse

DDPG and TD3 sparse are included too, but they are expected to be weaker.

## Comparable hyperparameters

The HER-family configs use the same shared settings:

- episodes: 800
- hidden_dim: 256
- replay_size: 1,000,000
- batch_size: 256
- actor_lr: 0.0001
- critic_lr: 0.0003
- gamma: 0.98
- tau: 0.005
- alpha: 0.2
- start_steps: 1000
- updates_per_step: 1
- HER k: 4
- HER future offset: 1
- eval every: 10
- eval episodes: 20


## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Required packages:

- gymnasium
- gymnasium-robotics
- numpy
- pandas
- torch
- matplotlib
- imageio
- optuna, only needed for `train_baselines_optuna.py`

## Expected outputs

Examples:

- `results/her_regular/logs/sac_her/FetchReach_sparse/seed0/...csv`
- `results/her_regular/plots/sac_her/FetchReach_sparse/seed0/.../*.png`
- `results/her_regular/models/sac_her/FetchReach_sparse/seed0/...pt`
- `results/her_regular/results/her_results.csv`

After all HER-family runs finish, run the VS Code config `Plot HER family comparison`.
