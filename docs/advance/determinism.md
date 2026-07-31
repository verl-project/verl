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

```yaml
actor_rollout_ref:
  rollout:
    full_determinism: true
    seed: 42
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
```

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

> **Important:** `PYTHONHASHSEED` must be set **before the Python interpreter starts**. verl handles this automatically — it sets `PYTHONHASHSEED` from `rollout.seed` before `ray.init()` and propagates it to all Ray actors. Do NOT set it manually.

## Configuration Reference

| Parameter | Default | Scope | Description |
|-----------|---------|-------|-------------|
| `actor_rollout_ref.rollout.full_determinism` | `false` | Rollout | Enables deterministic rollout generation |
| `actor_rollout_ref.rollout.seed` | `42` | Rollout | Base seed; each replica uses `replica_rank + seed` |
| `actor_rollout_ref.actor.fsdp_config.full_determinism` | `false` | Actor | Enables deterministic PyTorch ops for actor |
| `actor_rollout_ref.ref.fsdp_config.full_determinism` | `false` | Ref model | Enables deterministic PyTorch ops for reference model |
| `reward.reward_model.rollout.full_determinism` | `false` | Reward model | Enables deterministic RM inference (forces `max_num_seqs=1`) |
| `reward.reward_model.rollout.seed` | `42` | Reward model | Base seed for RM vLLM server |

## How It Works

Determinism is enforced at five layers. All must be enabled for full E2E reproducibility:

### PyTorch-level

`enable_full_determinism(seed)` sets `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`, `FLASH_ATTENTION_DETERMINISTIC`, seeds all RNGs, calls `torch.use_deterministic_algorithms(True, warn_only=True)`, and disables cuDNN benchmarking. Applied in all training engine implementations.

### Environment propagation

`main_ppo.run_ppo()` sets three env vars before `ray.init()`:

- `PYTHONHASHSEED` — freezes Python hash() and dict ordering
- `VERL_FULL_DETERMINISM` — signals subprocesses to apply determinism
- `VLLM_BATCH_INVARIANT` — makes vLLM outputs independent of batch composition

These are forwarded to all Ray actors via `PPO_RAY_RUNTIME_ENV`.

### vLLM batch invariance + per-request seed

`VLLM_BATCH_INVARIANT=1` ensures vLLM outputs don't depend on which other requests are batched together. Each `generate()` call injects `SamplingParams.seed = replica_rank + config.seed` to reset RNG per request.

### Priority scheduling + deterministic routing

Without determinism, request order depends on arrival timing and server selection on dict iteration order. When `full_determinism=true`:

- Each sample gets a globally unique priority injected into the batch (`non_tensor_batch["priority"]`), so each rollout request is scheduled with a stable, distinct priority
- `SingleTurnAgentLoop` uses `request_id=f"det-{priority}"` instead of random UUID
- `GlobalRequestLoadBalancer` tie-breaking uses `hash(request_id) % len(candidates)` — deterministic with frozen `PYTHONHASHSEED`

> **Note:** `priority` is a vLLM-only parameter. `LLMServerClient.generate()` automatically filters it for non-vLLM backends.

### Reward model serialization

vLLM's `/classify` endpoint (used by RM) does not support priority or batch invariance. When `full_determinism=true`, `RewardModelManager` forces `max_num_seqs=1`, serializing RM inference one request at a time.

## Side Effects and Limitations

**Performance**: deterministic PyTorch kernels are slower, cuDNN benchmarking is disabled, and RM `max_num_seqs=1` causes severe throughput loss. Typical E2E throughput drops 10–30% without RM; RM determinism can drop significantly more.

**Recommendation**: Only enable for debugging, regression testing, or research. Leave disabled for production training.

**Nondeterministic fallbacks**: Some GPU ops have no deterministic implementation. `torch.use_deterministic_algorithms(True, warn_only=True)` warns when these are encountered.

**Backend support**:

| Backend | Rollout | Reward model |
|---------|---------|--------------|
| vLLM | ✅ | ✅ (serialized) |
| SGLang | ❌ | ❌ |
| TensorRT-LLM | ❌ | ❌ |

**PYTHONHASHSEED**: Must be set before process start. verl handles this automatically; manual Ray actor creation must propagate it via `runtime_env`.

**Data parallelism**: Each replica uses `replica_rank + seed`, producing different but internally reproducible outputs. Two runs with the same config produce bitwise-aligned results.

**Multi-turn agent not supported**: Full determinism only works for single-turn rollouts (`single_turn_agent_loop`). Multi-turn rollouts (`tool_agent_loop`) are **not** bitwise reproducible, for two reasons:

- `tool_agent_loop` uses a random UUID per trajectory as `request_id` and does not pass `priority` to `server_manager.generate()`, so the deterministic routing and priority scheduling described above do not apply.
- Even with those added, each turn is interleaved with external tool calls whose execution time varies across runs, so the order in which requests arrive at vLLM cannot be made deterministic. This is inherent to multi-turn agentic workloads.

For bitwise-reproducible rollouts, use `single_turn_agent_loop`. (The per-request sampling seed is still applied to multi-turn requests inside the vLLM server, but this alone is not sufficient for end-to-end reproducibility.)

## Verifying Determinism

Rollout determinism (bitwise reproducible vLLM generation):

```bash
VLLM_DETERMINISM_DENSE_MODEL_PATH=${HOME}/models/Qwen/Qwen2.5-0.5B-Instruct \
VLLM_DETERMINISM_N_GPUS=2 \
pytest tests/workers/rollout/rollout_vllm/test_vllm_generation_determinism.py -v -s
```

E2E training (bitwise-aligned reward curves across two full PPO runs):

```bash
python tests/experimental/reward_loop/run_determinism_e2e_with_rm.py \
  --policy_model ~/models/Qwen/Qwen2.5-0.5B-Instruct \
  --rm_model ~/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B \
  --train_files ~/data/gsm8k/train.parquet \
  --val_files ~/data/gsm8k/test.parquet \
  --n_gpus 2 --n_steps 5
```
