# Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a family of policy gradient methods for reinforcement learning, proposed by OpenAI in 2017. PPO strikes a balance between simplicity, stability, and performance, making it one of the most widely used algorithms in modern RL applications, including large-scale language model fine-tuning.

Traditional policy gradient methods like REINFORCE or Vanilla Policy Gradient suffer from:

- High variance and sample inefficiency.
- Instability due to large policy updates.

PPO addresses this problem using a clipped surrogate objective that avoids overly large updates without requiring second-order derivatives.

For more technical details regarding PPO, we suggest reading the introduction in the [OpenAI spinning up tutorial](https://spinningup.openai.com/en/latest/algorithms/ppo.html), and the paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

## Key Components

- Actor-Critic Architecture: PPO requires both an actor model (policy) and a critic model (value function). This differs from other algorithms like GRPO and RLOO that don't require a critic model.

- Generalized Advantage Estimation (GAE): PPO uses GAE for computing advantage values, which helps reduce variance in policy gradient estimates while maintaining low bias.

- Clipped Surrogate Objective: The core of PPO is implemented through the clipped surrogate objective function that limits policy updates.

## Configuration

Note that all configs containing `micro_batch_size` are used to configure the maximum sample or token count per forward or backward pass to avoid GPU OOMs, whose value should not change algorithmic/convergence behavior.

Most critic configs are similar to those of actors. Note that the critic model is omitted from the figure below.

![image](https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d)

- `data.train_batch_size`: The global batch size of prompts used to generate a set of sampled trajectories/rollouts. The number of responses/trajectories is `data.train_batch_size * actor_rollout.ref.rollout.n`

- `actor_rollout_ref.actor.ppo_mini_batch_size`: The set of sampled trajectories is split into multiple mini-batches with batch_size=ppo_mini_batch_size for PPO actor updates. The ppo_mini_batch_size is a global size across all workers

- `critic.ppo_mini_batch_size`: The set of sampled trajectories is split into multiple mini-batches with batch_size=ppo_mini_batch_size for PPO critic updates. The ppo_mini_batch_size is a global size across all workers

- `actor_rollout_ref.actor.clip_ratio`: The PPO clip range. Default to 0.2

- `actor_rollout_ref.actor.ppo_epochs`: Number of epochs for PPO updates on one set of sampled trajectories for actor

- `critic.ppo_epochs`: Number of epochs for PPO updates on one set of sampled trajectories for critic. Defaults to `actor_rollout_ref.actor.ppo_epochs`

- `algorithm.gamma`: discount factor

- `algorithm.lam`: The lambda term that trades off between bias and variance in the GAE estimator

- `algorithm.adv_estimator`: Support gae, grpo, reinforce_plus_plus, reinforce_plus_plus_baseline, rloo, rloo_vectorized

## Advanced Extensions

### KL Divergence Control

Options to prevent the policy from diverging too far from a reference policy. Two mechanisms are available: KL reward penalty and KL loss. For more technical details, see [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

Options to use KL loss for KL divergence control: 

- `actor_rollout_ref.actor.use_kl_loss`: to use kl loss in the actor. When used, we are not applying KL in the reward function. Default is False

- `actor_rollout_ref.actor.kl_loss_coef`: The coefficient of kl loss. Default is 0.001.

- `actor_rollout_ref.actor.kl_loss_type`: Support kl(k1), abs, mse(k2), low_var_kl(k3) and full. Appending "+" in the end (e.g., 'k1+' and 'k3+') would apply straight through to employ k2 for unbiased gradient estimation, regardless of the kl value estimation (see https://github.com/volcengine/verl/pull/2953#issuecomment-3162113848 for more details). How to calculate the kl divergence between actor and reference policy. See this blog post for detailed analysis: http://joschu.net/blog/kl-approx.html

Options to use KL penalty in the reward:

- `algorithm.use_kl_in_reward`: Whether to enable in-reward kl penalty. Default is False.

- `algorithm.kl_penalty`: Support kl(k1), abs, mse(k2), low_var_kl(k3) and full. This defines the way to calculate the kl divergence between actor and reference policy. For specific options, refer to `kl_penalty` in core_algos.py. See this blog post for detailed analysis: http://joschu.net/blog/kl-approx.html

- `algorithm.kl_ctrl.kl_coef`: The (initial) coefficient of in-reward kl_penalty. Default is 0.001.
- `algorithm.kl_ctrl.type`: 'fixed' for FixedKLController and 'adaptive' for AdaptiveKLController.
- `algorithm.kl_ctrl.horizon`: See source code of AdaptiveKLController for details.
- `algorithm.kl_ctrl.target_kl`: See source code of AdaptiveKLController for details.

### Dual-clip PPO

The Dual-Clip PPO introduces a approach by applying a lower bound to the policy ratio when the advantage is less than zero, when multiplied by a large raito, does not exceed a specified lower bound.

![image](https://github.com/user-attachments/assets/fc232181-d8b0-4307-8dd2-4dc0a4c1c139)

- `actor_rollout_ref.actor.clip_ratio_c`: lower bound of the value for Dual-clip PPO, defaults to 3.0

## Reference Example

Qwen2.5 training log and commands: [link](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log)

```bash
bash run_gemma.sh
  trainer.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  trainer.logger=console \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=2 \
  critic.ppo_micro_batch_size=2
```

Reference performance with verl v0.2:

| Model                          | Method          | Score | Link                                                                                           |
|-------------------------------|------------------|-------|------------------------------------------------------------------------------------------------|
| Qwen/Qwen2.5-0.5B-Instruct     | pretrained model | 36.4  | [Qwen Blog](https://qwenlm.github.io/blog/qwen2.5-llm/)                                        |
| Qwen/Qwen2.5-0.5B-Instruct     | PPO              | 56.7  | [PPO Command and Logs](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log) |

## DeepSpeed Example

For a minimal DeepSpeed PPO command path (ZeRO-1/2/3 + optional offload), see:

```bash
bash examples/ppo_trainer/run_qwen3-0.6b_ds_gsm8k.sh
```
