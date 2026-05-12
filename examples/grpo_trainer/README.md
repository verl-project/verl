# Group Relative Policy Optimization (GRPO)

In reinforcement learning, classic algorithms like PPO rely on a "critic" model to estimate the value of actions, guiding the learning process. However, training this critic model can be resource-intensive.

GRPO simplifies this process by eliminating the need for a separate critic model. Instead, it operates as follows:
- Group Sampling: for a given problem, the model generates multiple possible solutions, forming a "group" of outputs.
- Reward Assignment: each solution is evaluated and assigned a reward based on its correctness or quality.
- Baseline Calculation: the average reward of the group serves as a baseline.
- Policy Update: the model updates its parameters by comparing each solution's reward to the group baseline, reinforcing better-than-average solutions and discouraging worse-than-average ones.

For more details, refer to the original paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300).

## Key Components

- No Value Function (Critic-less): unlike PPO, GRPO does not train a separate value network (critic).
- Group Sampling (Grouped Rollouts): instead of evaluating one rollout per input, GRPO generates multiple completions (responses) from the current policy for each prompt. This set of completions is referred to as a group.
- Relative Rewards: within each group, completions are scored (e.g., based on correctness), and rewards are normalized relative to the group.

## Important knobs

- `actor_rollout_ref.rollout.n`: per-prompt sample count (required >= 2 for GRPO).
- `data.train_batch_size`: prompts per global step. Total trajectories = `train_batch_size * rollout.n`.
- `actor_rollout_ref.actor.ppo_mini_batch_size`: global mini-batch for actor updates (must divide `train_batch_size * n`).
- `actor_rollout_ref.actor.ppo_epochs`: inner-loop epochs over the sampled trajectories.
- `actor_rollout_ref.actor.clip_ratio`: PPO clip range, default `0.2`.
- `actor_rollout_ref.actor.loss_agg_mode`: `token-mean` (default), `seq-mean-token-sum`, or `seq-mean-token-mean`.
- `actor_rollout_ref.actor.use_kl_loss=True` + `actor_rollout_ref.actor.kl_loss_coef` / `kl_loss_type`: regularise toward the reference policy via KL loss on the actor.
- `algorithm.adv_estimator=grpo`.

## Dr. GRPO

To enable Dr. GRPO (see [Understanding R1-Zero-Like Training](https://arxiv.org/pdf/2503.20783)), set on top of the canonical GRPO overrides:

```
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
actor_rollout_ref.actor.use_kl_loss=False
algorithm.norm_adv_by_std_in_grpo=False
```

## Canonical scripts

All scripts in this directory follow the naming convention:

```
run_<model>_<train-backend>[_<platform-or-variant>].sh
```

Where:
- `<model>` is the canonical size for a model family
  (`qwen3_8b` for dense text, `qwen3_30b_a3b` for MoE, `qwen2_5_vl_7b` / `qwen3_vl_8b` for vision,
  `qwen3_235b_a22b` / `deepseek_v3_671b` for scale demos).
- `<train-backend>` ∈ {`fsdp`, `megatron`, `mindspeed`}.
- `<platform-or-variant>` is used only for hardware-specific variants such as `gb200`, `fp8`, `veomni`,
  or MindSpeed NPU scripts.
- `INFER_BACKEND` selects rollout backend inside scripts that support multiple choices
  (`vllm`, `sglang`, or `trtllm`).
- `DEVICE` selects GPU/NPU paths inside scripts that support both platforms.

Every script exposes the commonly tuned knobs as environment variables at the top, so you can run:

```bash
MODEL_PATH=Qwen/Qwen3-14B \
NNODES=2 NGPUS_PER_NODE=8 \
INFER_BACKEND=sglang ROLLOUT_N=8 TRAIN_BATCH_SIZE=2048 \
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
```

### Defaults

- `dynamic batch size` and `sequence balancing` are enabled by default on all scripts.
- Text LLM scripts train on `gsm8k + math` by default; vision scripts train on `geo3k`.
- Scale-demo scripts (235B, 671B) train on `dapo-math-17k` / `aime-2024`.

### Matrix

| Model family          | `vllm` | `sglang` | `trtllm` | Train backend   | Platforms |
| --------------------- | :----: | :------: | :------: | --------------- | --------- |
| Qwen3-8B (dense)      | ✓      | ✓        | ✓        | FSDP, Megatron  | nvidia, npu (FSDP + MindSpeed), `_gb200` variant |
| Qwen2.5-VL-7B         | ✓      | ✓        | ✓        | FSDP, Megatron  | nvidia    |
| Qwen3-VL-8B           | ✓      |          |          | FSDP, Megatron  | nvidia, npu (FSDP) |
| Qwen3-VL-30B-A3B      | ✓      |          |          | FSDP, Megatron  | nvidia, npu (FSDP) |
| Qwen3-VL-235B-A22B    | ✓      |          |          | Megatron        | nvidia    |
| Qwen3-30B-A3B (MoE)   | ✓      | ✓        | ✓        | FSDP, Megatron  | nvidia, npu (MindSpeed) |
| Qwen3-235B-A22B       | ✓      |          | ✓        | Megatron        | nvidia, npu |
| Qwen3-Next-80B-A3B    | ✓      |          |          | FSDP            | npu       |
| Qwen3.5-27B (dense)   | ✓      |          |          | FSDP2           | nvidia, npu |
| Qwen3.5-35B (dense)   | ✓      |          |          | FSDP2, Megatron | nvidia, npu |
| Qwen3.5-35B-A3B (MoE) |        | ✓        |          | VeOmni          | nvidia    |
| Qwen3.5-122B-A10B     | ✓      |          |          | Megatron        | nvidia    |
| DeepSeek-V3 671B      | ✓      |          |          | Megatron        | nvidia    |
| GLM-4.1V-9B           | ✓      |          |          | FSDP            | nvidia    |
| MiniCPM-o-2.6         | ✓      |          |          | FSDP            | nvidia    |
| Moonlight-16B-A3B     | ✓      |          |          | Megatron        | nvidia    |
| Nemotron-Nano-v3-30B-A3B | ✓   |          |          | Megatron        | nvidia    |
| Seed-OSS-36B          | ✓      |          |          | FSDP2           | nvidia    |
| GPT-OSS-20B           |        | ✓        |          | FSDP            | nvidia    |
| Mistral-Nemo-12B (RM demo) | ✓ |          |          | FSDP            | nvidia    |

LoRA variants live in `examples/tuning/lora/`, profiling variants in `examples/profile/`.
Scale / hardware-specific demos (e.g. `run_qwen3_8b_fsdp_gb200.sh`, FP8 variants, VeOmni) keep a trailing suffix to stay discoverable.

## Reference

- See [verl baselines](https://verl.readthedocs.io/en/latest/algo/baseline.html) for reference metrics.
- Qwen2.5 GRPO training log: [experiments/gsm8k/qwen2-7b-fsdp2.log](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b-fsdp2.log).
