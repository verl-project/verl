# Training Flow Matching Models via Online RL (Flow-GRPO)

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
| **KL regularization** | KL penalty added to reward or directly to loss | KL loss applied to SDE steps; `use_kl_loss=True` recommended |
| **CFG (guidance)** | Not applicable | CFG distillation occurs naturally; CFG can be disabled at both train and test time |
| **Advantage estimator** | `algorithm.adv_estimator=grpo` | `algorithm.adv_estimator=flow_grpo` |
| **Loss mode** | `actor_rollout_ref.actor.policy_loss.loss_mode` not diffusion-specific | `actor_rollout_ref.actor.policy_loss.loss_mode=flow_grpo` |

## Configuration

### Core parameters

- `algorithm.adv_estimator`: Set to `flow_grpo` (instead of `grpo`).

- `actor_rollout_ref.actor.policy_loss.loss_mode`: Set to `flow_grpo`.

- `actor_rollout_ref.rollout.n`: Number of image trajectories to sample per prompt for group-relative advantage computation. Analogous to GRPO's group size; should be > 1 (default in examples: `16`).

- `actor_rollout_ref.rollout.noise_level`: Controls the SDE noise injection level during rollout. Larger values increase diversity but may degrade image quality. Typical value: `1.2`.

- `actor_rollout_ref.rollout.sde_window_size`: Number of denoising steps to train on per trajectory (Denoising Reduction). Reducing this from the full step count speeds up training significantly.

- `actor_rollout_ref.rollout.sde_window_range`: The range of denoising steps from which the training window is sampled, e.g., `[0, 5]` to focus on early (high-noise) steps.

- `actor_rollout_ref.rollout.val_kwargs.num_inference_steps`: Full number of denoising steps used during inference/evaluation. This is kept at its original value (e.g., `50`) and is independent of `sde_window_size`.

- `actor_rollout_ref.rollout.guidance_scale`: Classifier-free guidance scale during rollout. Can be set to `1.0` (no CFG) because the RL process naturally performs CFG distillation.

- `actor_rollout_ref.actor.use_kl_loss`: Set to `True` to add a KL divergence term between the trained policy and the reference policy to the loss.

- `actor_rollout_ref.actor.kl_loss_coef`: Coefficient for the KL loss term.

## Data Preprocessing

All training scripts expect the dataset in parquet format. The examples use an OCR dataset from the [Flow-GRPO repository](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr). The raw dataset consists of text files where each ground-truth answer is stored in the format `The image displays "xxx".`. Before running any training script, convert it to parquet format using the provided preprocessing script.

### Step 1: Download the raw dataset

Download the OCR dataset from the Flow-GRPO repository and place it at `~/dataset/ocr/` (or any path of your choice):

```bash
# Clone or download from https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr
# Place the dataset directory at ~/dataset/ocr/
# Expected structure:
#   ~/dataset/ocr/
#       train/   (or train split files)
#       test/    (or test split files)
```

### Step 2: Run the preprocessing script

```bash
python examples/data_preprocess/qwenimage_ocr.py \
    --local_dataset_path ~/dataset/ocr \
    --local_save_dir ~/data/ocr
```

The output parquet files are consumed directly by all training scripts via `data.train_files` and `data.val_files`.

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

### Full Fine-Tuning

To fine-tune all model weights instead of using LoRA:

```bash
bash examples/flowgrpo_trainer/run_flowgrpo_full_ft.sh
```

## Reference Example

Standard LoRA training with OCR reward (Qwen-Image, 4 GPUs) with CFG and KL loss enabled:

```bash
bash examples/flowgrpo_trainer/run_flowgrpo.sh
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
