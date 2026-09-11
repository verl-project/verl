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

trainer:
  # REQUIRED: use the V0 trainer backend. The default V1 backend collects rollout
  # outputs in completion order (via a TransferQueue), which varies across runs and
  # breaks bitwise reproducibility. V0 uses asyncio.gather (submission order).
  use_v1: false
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
  trainer.use_v1=false \
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
| `trainer.use_v1` | `true` | Trainer | Must be `false` under `full_determinism`. V1 (AgentLoopManagerTQ) collects outputs in completion order, which varies across runs; V0 (AgentLoopManager) uses submission order. |

## How It Works

`full_determinism=true` is enforced at five layers. The training entrypoint sets `PYTHONHASHSEED` (from `rollout.seed`), `VERL_FULL_DETERMINISM`, `VLLM_BATCH_INVARIANT`, and the NCCL/cuBLAS determinism env vars (`CUBLAS_WORKSPACE_CONFIG`, `FLASH_ATTENTION_DETERMINISTIC`, `NCCL_DETERMINISTIC`, `NCCL_ALGO`) before `ray.init()` and forwards them to all Ray actors via `PPO_RAY_RUNTIME_ENV`. These must be set before torch/NCCL init, so the entrypoint exports them rather than `enable_full_determinism()` (which runs after the actor is already up). Do NOT set `PYTHONHASHSEED` manually — the entrypoint handles it.

### Floating-point determinism

`enable_full_determinism(seed)` sets `CUBLAS_WORKSPACE_CONFIG`, `FLASH_ATTENTION_DETERMINISTIC`, `NCCL_DETERMINISTIC`, `NCCL_ALGO`, seeds all RNGs, calls `torch.use_deterministic_algorithms(True, warn_only=True)`, and disables cuDNN benchmarking. Applied in all training engine implementations (FSDP, Megatron, etc.). The flash-attn Triton cross-entropy kernel (used to compute log-probs) has a non-deterministic reduction not covered by the env vars above, so `VERL_DISABLE_FLASH_ATTN_CE=1` is set by default to force the pure-PyTorch `log_softmax`+gather path.

### Sampling seeds

- **Replica seed**: each replica uses `replica_rank + config.seed`, producing different but internally reproducible outputs across replicas. Two runs with the same config produce bitwise-aligned results.
- **Per-request seed**: each `generate()` call injects `SamplingParams.seed = replica_rank + config.seed` to reset the sampler RNG per request, so the same prompt+seed yields the same tokens regardless of batch.

### Deterministic routing

- **Actor rollout**: `SingleTurnAgentLoop` uses `request_id=f"det-{priority}"` (priority from `non_tensor_batch["priority"]`), and `GlobalRequestLoadBalancer` (in `verl/workers/rollout/router.py`, with `full_determinism=True`) routes with `hash(request_id) % len(servers)` — the same request always routes to the same vLLM server across runs. (`priority` is vLLM-only; `LLMServerClient.generate()` filters it for non-vLLM backends.) When `full_determinism=True`, `LLMServerClient._vllm_request_id()` forwards this stable `request_id` to vLLM itself.
- **Reward**: `NaiveRouter` routes with `binascii.crc32(request body) % len(candidates)` among equally-loaded RM replicas, so the same reward request always routes to the same replica. This neutralizes replica-level floating-point differences that seed alone cannot equalize.

### Trainer backend (V0 required)

Full determinism requires the **V0** trainer backend (`trainer.use_v1=false`, i.e. `AgentLoopManager`). The default V1 backend (`AgentLoopManagerTQ`) decouples rollout submission from collection via a fire-and-forget TransferQueue: outputs are collected **in completion order**, which depends on per-request latency and varies across runs, so the output concatenation order (and thus the training batch) differs run-to-run. V0 instead uses `asyncio.gather`, which returns outputs in **submission order** — stable across runs.

Set `trainer.use_v1=false` explicitly when enabling `full_determinism`.

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
- **Trainer backend**: the V0 trainer (`trainer.use_v1=false`) is required. The default V1 backend's TransferQueue collects outputs in completion order, which is nondeterministic (see [Trainer backend](#trainer-backend-v0-required) above).
- **Multi-turn agent**: not supported. Full determinism only works for single-turn rollouts (`single_turn_agent_loop`). Multi-turn rollouts (`tool_agent_loop`) are **not** bitwise reproducible — `tool_agent_loop` uses a random UUID per trajectory as `request_id`, does not pass `priority`, and each turn is interleaved with external tool calls whose timing varies across runs. Use `single_turn_agent_loop` for bitwise-reproducible rollouts.

## Verifying Determinism

Rollout determinism (bitwise reproducible vLLM generation):

```bash
VLLM_DETERMINISM_DENSE_MODEL_PATH=${HOME}/models/Qwen/Qwen2.5-0.5B-Instruct \
VLLM_DETERMINISM_N_GPUS=2 \
pytest tests/workers/rollout/rollout_vllm/test_vllm_generation_determinism.py -v -s
```

E2E training (bitwise-aligned reward curves across two full PPO runs). Runs PPO twice with identical seeds, compares per-step reward curves in float32; exit code 0 = aligned, 1 = not aligned / error (usable as a CI gate):

```bash
python tests/experimental/reward_loop/run_determinism_e2e_with_rm.py \
  --policy_model ~/models/Qwen/Qwen2.5-0.5B-Instruct \
  --rm_model ~/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B \
  --train_files ~/data/gsm8k/train.parquet \
  --val_files ~/data/gsm8k/test.parquet \
  --n_gpus 2 --n_steps 2
```

See the script's module docstring for the full argument reference (e.g. `--plot`, `--save_metrics`, multi-replica coverage, and why the RM `max_num_seqs` is forced to 1 for discriminative RMs but allowed `>1` for generative RMs).
