# Flow-GRPO

Last updated: 04/22/2026.

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

- `actor_rollout_ref.rollout.external_lib`: Python module path imported on
  every rollout worker before the engine starts. Use this to register custom
  pipeline implementations (e.g., `examples.flowgrpo_trainer.vllm_omni_impl`
  for the Qwen-Image `vllm_omni` example). The module must call
  `@VllmOmniPipelineBase.register(...)` at import time.

#### Model

- `actor_rollout_ref.model.path`: Base diffusion model path.

- `actor_rollout_ref.model.tokenizer_path`: Optional tokenizer path if it is
  not located under the model path.

#### Batch size

FlowGRPO uses three nested batch-size parameters that operate at different
stages of the training loop. They address different concerns (RL sample
diversity, multi-epoch reuse, and GPU memory) and must be understood together.

**Step 1 — Rollout (`data.train_batch_size`)**

`data.train_batch_size` is the number of **unique prompts** drawn from the
dataset per training step. Before rollout, each prompt is replicated
`actor_rollout_ref.rollout.n` times so that the rollout engine generates `n`
independent image trajectories per prompt. The in-memory batch after rollout
therefore holds `train_batch_size × n` image samples. GRPO advantage
normalization runs over this **full** batch — it needs all `n` trajectories
for every prompt to compute group-relative rewards before any splitting occurs.

**Step 2 — Actor update (`actor_rollout_ref.actor.ppo_mini_batch_size`)**

`ppo_mini_batch_size` controls how the full post-rollout batch is sliced for
actor gradient updates. **Important:** this value is specified in **prompts**,
not image samples. The trainer internally scales it by `rollout.n` to get
the actual mini-batch size in samples:

```
effective mini-batch = ppo_mini_batch_size × rollout.n  (image samples)
number of mini-batches per epoch = train_batch_size / ppo_mini_batch_size
```

All `n` trajectories belonging to the same prompt are kept in the same
mini-batch. This is not optional: although advantages are already computed
globally before this split, the gradient update for each image depends on its
advantage relative to the other images in its group. Scattering a prompt's
trajectories across different mini-batches would break that correspondence.
`ppo_mini_batch_size` must divide `train_batch_size` evenly.

**Step 3 — FSDP sharding and gradient accumulation
(`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`)**

Each mini-batch is distributed across GPUs by FSDP data parallelism, so each
GPU receives `(ppo_mini_batch_size × n) / n_gpus` image samples. That
per-GPU shard is then **chunked into micro-batches** of
`ppo_micro_batch_size_per_gpu` for the actual forward/backward passes, with
gradients accumulated across chunks before the optimizer step. This is pure
gradient accumulation: the effective gradient is identical to running the full
per-GPU shard in one shot; only peak activation memory changes.

For diffusion models the accumulation is two-dimensional: the engine also
loops over each active denoising timestep inside every micro-batch, so the
total gradient accumulation steps per GPU per mini-batch is:

```
gradient_accumulation_steps = (per_gpu_samples / ppo_micro_batch_size_per_gpu)
                              × sde_window_size
```

`ppo_micro_batch_size_per_gpu` must satisfy:
`(ppo_mini_batch_size × n) / n_gpus` is divisible by
`ppo_micro_batch_size_per_gpu`.

**Concrete walkthrough** (reference OCR script, 4 GPUs, `sde_window_size=2`):

```
data.train_batch_size              = 32    # 32 prompts loaded
actor_rollout_ref.rollout.n        = 16    # 16 images generated per prompt
  → post-rollout batch             = 512   # advantage computed over all 512

ppo_mini_batch_size (config)       = 16    # in prompts
  → effective mini-batch           = 16 × 16 = 256 samples
  → mini-batches per epoch         = 512 / 256 = 2 actor gradient steps

FSDP shards 256 samples across 4 GPUs:
  → per-GPU samples                = 256 / 4 = 64

ppo_micro_batch_size_per_gpu       = 16
  → micro-batches per GPU          = 64 / 16 = 4
  → gradient_accumulation_steps    = 4 × 2 (sde_window_size) = 8
```

#### Reward

- `reward.reward_manager.name`: Selects the reward manager.

- `reward.custom_reward_function.path` and
  `reward.custom_reward_function.name`: Register the task-specific reward
  post-processing function such as `compute_score_ocr`.

For an end-to-end OCR training walkthrough, including dataset preparation and
the full runnable command, see `docs/start/flowgrpo_quickstart.rst`.


## Reference Example

Standard LoRA training with OCR reward (Qwen-Image, 4 GPUs) using the current
`vllm_omni` rollout example:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh
```

## Variants

### Rule-Based Reward Training: JPEG incompressibility

FlowGRPO also supports rule-based rewards that score images directly without a
VLM reward model, using the same `reward.reward_manager.name=visual` setup.

`verl/utils/reward_score/jpeg_compressibility.py` rewards images that are
harder to JPEG-compress (richer texture, more complex content). No extra
dependencies or reward model process are required.

Minimal dataset row:

```python
{
    "data_source": "jpeg_compressibility",
    "prompt": [{"role": "user", "content": "<your prompt>"}],
    "reward_model": {"ground_truth": ""},  # required by schema, ignored by scorer
}
```

Config changes relative to the OCR example — **remove** these lines:

```bash
reward.reward_model.enable=True
reward.reward_model.model_path=...
reward.reward_model.rollout.name=...
reward.reward_model.rollout.tensor_model_parallel_size=...
reward.custom_reward_function.path=...
reward.custom_reward_function.name=...
```

Keep `reward.reward_manager.name=visual` and all actor/rollout settings
unchanged.

### Async Reward


For reward models that are expensive to evaluate (e.g., a VLM judge), the reward model can be allocated its own dedicated GPU resource pool and run asynchronously alongside the policy. This avoids blocking policy training on reward computation.

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_async_reward.sh
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