# Flow-GRPO

Last updated: 04/15/2026.

Flow-GRPO ([paper](https://arxiv.org/abs/2505.05470), [code](https://github.com/yifan123/flow_grpo)) is the first method to integrate online policy gradient reinforcement learning into **flow matching** generative models (e.g., Stable Diffusion 3, FLUX). It enables direct reward optimization for tasks such as compositional text-to-image generation, visual text rendering, and human preference alignment, without modifying the standard inference pipeline.

Two core technical contributions make this possible:

1. **ODE-to-SDE Conversion**: Flow matching models natively use a deterministic ODE sampler. Flow-GRPO converts this ODE into an equivalent SDE that preserves the model's marginal distribution at every timestep. This introduces the stochasticity required for group sampling and RL exploration.

2. **Denoising Reduction**: Training on all denoising steps is expensive. Flow-GRPO reduces the number of *training* steps while keeping the original number of *inference* steps, significantly improving sampling efficiency without sacrificing reward performance.

Empirically, RL-tuned SD3.5-M with Flow-GRPO raises GenEval accuracy from 63% to 95% and visual text rendering accuracy from 59% to 92%.

## Key Components

- **Flow Matching Backbone**: operates on continuous-time flow matching models (e.g., SD3.5, FLUX) rather than discrete-token LLMs.
- **ODE-to-SDE Rollout**: generates a group of diverse image trajectories by injecting controlled noise via SDE sampling at selected denoising steps.
- **Denoising Reduction**: trains on a reduced subset of denoising steps (configurable via `sde_window_size` and `sde_window_range`) while inference uses the full step count.
- **Image Reward Models**: rewards are assigned by external reward models (e.g., GenEval, OCR, PickScore, aesthetic score) rather than rule-based verifiers.
- **No Critic**: like GRPO for LLMs, no separate value network is trained; advantages are computed from group-relative rewards.

## Key Differences: GRPO vs. Flow-GRPO

| Dimension | GRPO (LLM) | Flow-GRPO (Diffusion) |
|---|---|---|
| **Model type** | Autoregressive language model | Flow matching / diffusion model |
| **Action space** | Discrete token sequences | Continuous denoising trajectories (SDE paths) |
| **Rollout mechanism** | Sample `n` token sequences per prompt | Convert ODE to SDE; sample `n` image trajectories per prompt via stochastic denoising |
| **Log-probability** | Standard next-token log-prob | Log-prob of the SDE noise prediction at each selected denoising step |
| **Training steps** | All decoding steps are trivially identical in cost | Denoising Reduction: train on a small window of steps, infer with full steps |
| **Reward signal** | Rule-based verifiers or LLM judges on text | Image reward models (GenEval, OCR, PickScore, aesthetic, etc.) |
| **KL regularization** | KL penalty added to reward or directly to loss | KL-style regularization is available, but the exact setup depends on the training config |
| **CFG (guidance)** | Not applicable | CFG distillation occurs naturally; CFG can be disabled at both train and test time |
| **Advantage estimator** | `algorithm.adv_estimator=grpo` | `algorithm.adv_estimator=flow_grpo` |
| **Loss mode** | `actor_rollout_ref.actor.policy_loss.loss_mode` not diffusion-specific | `actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo` |

## Configuration

Diffusion training now uses dedicated diffusion config blocks. In `verl/trainer/config/diffusion_trainer.yaml`,
the main sections are:

- `algorithm`: diffusion-specific advantage computation and normalization
- `actor_rollout_ref.actor`: optimization and diffusion loss settings
- `actor_rollout_ref.rollout`: rollout backend, sampling, and SDE controls
- `actor_rollout_ref.model`: model path plus diffusion-model / LoRA settings
- `reward`: reward manager, reward model, and custom reward function

The default diffusion model YAML mirrors several rollout fields
(`num_inference_steps`, `true_cfg_scale`, `max_sequence_length`,
`guidance_scale`, and `algo`) into `actor_rollout_ref.model.*`, so in practice
the rollout section is the main place to override sampling behavior.

### Core parameters

#### Algorithm

- `algorithm.adv_estimator`: Set to `flow_grpo`.

#### Actor / loss

- `actor_rollout_ref.actor.diffusion_loss.loss_mode`: Set to `flow_grpo`.

- `actor_rollout_ref.actor.diffusion_loss.clip_ratio`: clipping
  factor used in the diffusion loss.

- `actor_rollout_ref.actor.diffusion_loss.adv_clip_max`: Maximum absolute
  advantage used before computing the policy loss.

- `actor_rollout_ref.actor.use_kl_loss`: Enables KL loss against the reference
  policy.

- `actor_rollout_ref.actor.kl_loss_coef`: Coefficient for the KL term when KL enabled.

#### Rollout / sampling

- `actor_rollout_ref.rollout.name`: Selects the rollout backend. Currently supports `vllm_omni`.

- `actor_rollout_ref.rollout.n`: Number of sampled image trajectories per
  prompt. This is the FlowGRPO group size and should be greater than `1`.

- `actor_rollout_ref.rollout.algo.noise_level`: Magnitude of SDE noise injected
  during rollout. Larger values increase diversity but can hurt image quality.

- `actor_rollout_ref.rollout.algo.sde_type`: SDE variant for rollout. The
  current example uses `sde`.

- `actor_rollout_ref.rollout.algo.sde_window_size`: Number of denoising steps
  included in the active training window. Smaller values reduce training cost.

- `actor_rollout_ref.rollout.algo.sde_window_range`: Range used to sample the
  start of that active denoising window.

- `actor_rollout_ref.rollout.num_inference_steps`: Number of denoising steps
  used for rollout generation during training.

- `actor_rollout_ref.rollout.val_kwargs.num_inference_steps`: Number of
  denoising steps used during validation / evaluation.

- `actor_rollout_ref.rollout.true_cfg_scale`: True classifier-free guidance
  scale used during rollout. Used in `Qwen-Image`.

- `actor_rollout_ref.rollout.guidance_scale`: Distilled guidance scale for
  models that expose a guidance embedding; keep `null` to disable it.

- `actor_rollout_ref.rollout.engine_kwargs.vllm_omni.custom_pipeline`:
  Required by the `vllm_omni` Qwen-Image example to register the custom
  pipeline implementation.

#### Model

- `actor_rollout_ref.model.path`: Base diffusion model path.

- `actor_rollout_ref.model.tokenizer_path`: Optional tokenizer path if it is
  not located under the model path.

#### Reward

- `reward.reward_manager.name`: Selects the reward manager.

- `reward.custom_reward_function.path` and
  `reward.custom_reward_function.name`: Register the task-specific reward
  post-processing function such as `compute_score_ocr`.

For an end-to-end OCR training walkthrough, including dataset preparation and
the full runnable command, see `docs/start/flowgrpo_quickstart.rst`.

## Variants

### Flow-GRPO-Fast

Flow-GRPO-Fast accelerates training by confining stochasticity to only one or two denoising steps per trajectory:

1. Generate a deterministic ODE trajectory for each prompt.
2. At a randomly chosen intermediate step, inject noise and switch to SDE sampling to produce the group.
3. Continue the remaining steps with ODE sampling.

This significantly reduces training cost: only the selected step(s) require gradient computation, and sampling before the branching point does not need group expansion. Flow-GRPO-Fast with 2 training steps matches full Flow-GRPO reward performance.

```bash
bash examples/flowgrpo_trainer/run_flowgrpo_fast.sh
```

### Async Reward

For reward models that are expensive to evaluate (e.g., a VLM judge), the reward model can be allocated its own dedicated GPU resource pool and run asynchronously alongside the policy. This avoids blocking policy training on reward computation.

```bash
bash examples/flowgrpo_trainer/run_flowgrpo_async_reward.sh
```

## Reference Example

Standard LoRA training with OCR reward (Qwen-Image, 4 GPUs) using the current
`vllm_omni` rollout example:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh
```

## Citation

```bibtex
@article{liu2025flow,
  title={Flow-GRPO: Training Flow Matching Models via Online RL},
  author={Liu, Jie and Liu, Gongye and Liang, Jiajun and Li, Yangguang and Liu, Jiaheng and Wang, Xintao and Wan, Pengfei and Zhang, Di and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2505.05470},
  year={2025}
}
```