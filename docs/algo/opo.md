# On-Policy RL with Optimal Reward Baseline (OPO)

Last updated: 06/02/2025.

Loose on-policy constraints and suboptimal baselines in reinforcement learning often lead to training instability such as large policy shifts and entropy collapse. OPO addresses these challenges by using exact on-policy training with the theretically optimal reward baseline for advantage estimation. It achieves lower policy shifts and higher output entropy, encouraging more diverse and less repetitive responses.

OPO uses group sampling to generate multiple outputs for each input like GRPO. Unlike group-based algorithms which typically use the mean reward of a group as its baseline, OPO employs a theoretically optimal baseline: the length-weighted reward of the group. It also  omits the standard deviation normalization. By adopting these two key components, OPO enables the training of a single policy model with the objective of maximizing only the expected reward. For more detailes, refer to the original paper [On-Policy RL with Optimal Reward Baseline](https://arxiv.org/pdf/2505.23585).

## Key Components

- Exact On-Policy Training: always generates responses from the current policy, without using any pre-generated data or off-policy data.
- Optimal Reward Baseline: uses a length-weighted reward of the group as the baseline for normalizing the rewards.

## Configuration

To configure OPO within the framework, use the following YAML settings. These parameters are crucial for enabling exact on-policy training and activating the optimal reward baseline.

```yaml
algorithm:
  adv_estimator: opo  # Use OPO for optimal reward baseline 
data:
  train_batch_size: 1024
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 1024 # ppo_mini_batch_size should equal to train_batch_size to enable exact on-policy training
    entropy_coeff: 0 # disable entropy regularization
    use_kl_loss: False # disable kl regularization
    kl_loss_coef: 0 
```

## Advanced Extensions

OPO can also be extended to other algorithms like RLOO and Reinforce++. It just needs to adjust their configurations to enable exact on-policy training and incorporate the optimal length-weighted reward baseline with minimal modifications to their advantage estimation functions.
