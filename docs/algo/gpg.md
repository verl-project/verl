# GPG: Group Policy Gradient

Last updated: 07/03/2025.

Group Policy Gradient (GPG) is a minimalist reinforcement learning (RL) method that enhances the reasoning ability of large language models without relying on supervised fine-tuning or complex tricks. GPG revisits traditional policy gradients and directly optimizes the RL objective—no surrogate losses, no KL penalties, no critic, and no reference model. Compared to GRPO, GPG is simpler, more efficient, and achieves better results on many tasks. For more details, please refer to the original paper [GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning
](https://arxiv.org/abs/2504.02546).

## Key Components
- Use a corrected advantage function to improve policy gradient accuracy and training efficiency.
- By eliminating the critic and reference models, avoiding KL divergence constraints, significantly simplifies the training process compared to Group Relative Policy Optimization (GRPO)

## Configuration
To configure GPG within the framework, use the following YAML settings.

```yaml
algorithm:
  adv_estimator: gpg 
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: "gpg"
```

## Advanced Extensions
GPG is a simple and strong baseline for model reasoning. Although it avoids using KL loss in its original form, you can still use KL loss to further improve the performance.

```yaml
algorithm:
  adv_estimator: gpg
actor_rollout_ref:
  actor:
    use_kl_loss: True # enable kl regularization
    kl_loss_coef: 0.01
    policy_loss:
      loss_mode: "gpg"
```