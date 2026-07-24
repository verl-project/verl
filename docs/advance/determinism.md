# Full Determinism for Reproducible RL Training

**Authors**: Haichuan Hu, Yongxiang Huang, Jiawei Zhang, Nguyen Long

Last updated: 06/16/2026.

## Overview

By default, RL training in verl is **not** bitwise reproducible: identical configs run twice can produce different reward curves due to nondeterminism in GPU kernels, request scheduling, hash-based routing, and batch composition. The full determinism feature closes these gaps, enabling two identical runs to produce **bitwise-aligned reward curves**.

Useful for:

- **Debugging**: reproduce a training failure exactly, step-by-step
- **Regression testing**: verify that a code change has no silent effect on training outcomes
- **Research**: ensure fair comparison when evaluating algorithmic changes

## Quick Start

`full_determinism` is supported on all training engines (FSDP, Megatron, etc.) via each engine's config. The example below uses FSDP; for Megatron set `actor_rollout_ref.actor.megatron_config.full_determinism=true` (and similarly for the ref model).

```yaml
actor_rollout_ref:
  rollout:
    full_determinism: true
    seed: 42
    # If the policy model is not covered by vLLM batch invariance, set to 1.
    # max_num_seqs: 1
  actor:
    fsdp_config:
      full_determinism: true
  ref:
    fsdp_config:
      full_determinism: true

reward:
  reward_model:
    enable: true
    rollout:
      full_determinism: true
      seed: 42
      # If the RM model is not covered by vLLM batch invariance, set to 1.
      # max_num_seqs: 1
```

Both actor rollout and generative RM (VLM scoring) rely on vLLM batch invariance for co-batching. If your model or hardware is not covered, set `max_num_seqs=1` to serialize. Discriminative RM (score-outputting, e.g. Skywork-Reward) is forced to `1` automatically (see Reward model routing).

Or via Hydra overrides:

```bash
python -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.full_determinism=true \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.actor.fsdp_config.full_determinism=true \
  actor_rollout_ref.ref.fsdp_config.full_determinism=true \
  reward.reward_model.enable=true \
  reward.reward_model.rollout.full_determinism=true \
  reward.reward_model.rollout.seed=42 \
  [other config overrides...]
```

If the policy or RM model is not covered by vLLM batch invariance, add `actor_rollout_ref.rollout.max_num_seqs=1` and/or `reward.reward_model.rollout.max_num_seqs=1` to serialize. Discriminative RM (score-outputting) is forced to 1 automatically.

## Configuration Reference

| Parameter | Default | Scope | Description |
|-----------|---------|-------|-------------|
| `actor_rollout_ref.rollout.full_determinism` | `false` | Rollout | Enables deterministic rollout generation |
| `actor_rollout_ref.rollout.max_num_seqs` | `1024` | Rollout | Set to `1` to serialize if the policy model is not covered by vLLM batch invariance |
| `actor_rollout_ref.rollout.seed` | `42` | Rollout | Base seed; each replica uses `replica_rank + seed` |
| `actor_rollout_ref.actor.fsdp_config.full_determinism` | `false` | Actor | Enables deterministic PyTorch ops for actor |
| `actor_rollout_ref.ref.fsdp_config.full_determinism` | `false` | Ref model | Enables deterministic PyTorch ops for reference model |
| `reward.reward_model.rollout.full_determinism` | `false` | Reward model | Enables deterministic RM inference |
| `reward.reward_model.rollout.max_num_seqs` | `1024` | Reward model | Discriminative RM forced to 1 under full_determinism; set to 1 for generative RM if not covered by batch invariance |
| `reward.reward_model.rollout.seed` | `42` | Reward model | Base seed for RM vLLM server |

## How It Works

`full_determinism=true` is enforced at four layers. `main_ppo.run_ppo()` sets `PYTHONHASHSEED` (from `rollout.seed`, before the interpreter starts), `VERL_FULL_DETERMINISM`, `VLLM_BATCH_INVARIANT` before `ray.init()` and forwards them to all Ray actors via `PPO_RAY_RUNTIME_ENV`. Do NOT set `PYTHONHASHSEED` manually — verl handles it.

### Floating-point determinism

`enable_full_determinism(seed)` sets `CUBLAS_WORKSPACE_CONFIG`, `FLASH_ATTENTION_DETERMINISTIC`, seeds all RNGs, calls `torch.use_deterministic_algorithms(True, warn_only=True)`, and disables cuDNN benchmarking. Applied in all training engine implementations (FSDP, Megatron, etc.).

### Sampling seeds

- **Replica seed**: each replica uses `replica_rank + config.seed`, producing different but internally reproducible outputs across replicas. Two runs with the same config produce bitwise-aligned results.
- **Per-request seed**: each `generate()` call injects `SamplingParams.seed = replica_rank + config.seed` to reset the sampler RNG per request, so the same prompt+seed yields the same tokens regardless of batch.

### Deterministic routing

- **Actor rollout**: `SingleTurnAgentLoop` uses `request_id=f"det-{priority}"` (priority from `non_tensor_batch["priority"]`), and `GlobalRequestLoadBalancer` tie-breaks with `hash(request_id) % len(candidates)` — the same request always routes to the same vLLM server across runs. (`priority` is vLLM-only; `LLMServerClient.generate()` filters it for non-vLLM backends.)
- **Reward**: `NaiveRouter` tie-breaks equally-loaded RM replicas with `hash(request body) % len(candidates)`, so the same reward request always routes to the same replica. This neutralizes replica-level floating-point differences that seed alone cannot equalize.

### Batch invariance

`VLLM_BATCH_INVARIANT=1` makes vLLM outputs independent of batch composition. Coverage is model- and hardware-dependent — see the [vLLM batch invariance docs](https://docs.vllm.ai/en/latest/features/batch_invariance/) (and [tested models](https://docs.vllm.ai/en/latest/features/batch_invariance/#tested-models)). If not covered, set `max_num_seqs=1` to serialize.

For reward specifically:
- **Discriminative RM** (score-outputting, e.g. Skywork-Reward; no custom reward fn): `max_num_seqs` is **forced to 1** — batch invariance is verified on generation models, not score-outputting RM architectures.
- **Generative RM** (VLM that outputs text scores, via custom reward fn): `max_num_seqs` is **not forced** — user-managed; rely on batch invariance + per-request seed.

## Side Effects

- **Performance**: deterministic PyTorch kernels are slower and cuDNN benchmarking is disabled. Discriminative RM is serialized (`max_num_seqs=1`) under full_determinism.
- **Recommendation**: Only enable for debugging, regression testing, or research. Leave disabled for production training.

## Limitations

- **Hardware**: vLLM batch invariance (and some deterministic GPU ops) requires specific hardware — see the [vLLM batch invariance docs](https://docs.vllm.ai/en/latest/features/batch_invariance/) for requirements. On unsupported hardware, set `max_num_seqs=1` to serialize. `torch.use_deterministic_algorithms(True, warn_only=True)` warns when a deterministic kernel is unavailable.
- **Backend**: only vLLM is supported.
- **Multi-turn agent**: not supported. Full determinism only works for single-turn rollouts (`single_turn_agent_loop`). Multi-turn rollouts (`tool_agent_loop`) are **not** bitwise reproducible — `tool_agent_loop` uses a random UUID per trajectory as `request_id`, does not pass `priority`, and each turn is interleaved with external tool calls whose timing varies across runs. Use `single_turn_agent_loop` for bitwise-reproducible rollouts.

## Verifying Determinism

Rollout determinism (bitwise reproducible vLLM generation):

```bash
VLLM_DETERMINISM_DENSE_MODEL_PATH=${HOME}/models/Qwen/Qwen2.5-0.5B-Instruct \
VLLM_DETERMINISM_N_GPUS=2 \
pytest tests/workers/rollout/rollout_vllm/test_vllm_generation_determinism.py -v -s
```

E2E training (bitwise-aligned reward curves across two full PPO runs). The discriminative RM path goes through `NaiveRouter` and is serialized (`max_num_seqs=1`):

```bash
python tests/experimental/reward_loop/run_determinism_e2e_with_rm.py \
  --policy_model ~/models/Qwen/Qwen2.5-0.5B-Instruct \
  --rm_model ~/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B \
  --train_files ~/data/gsm8k/train.parquet \
  --val_files ~/data/gsm8k/test.parquet \
  --n_gpus 2 --rm_tp 1 --rollout_tp 1 --n_steps 10
```
