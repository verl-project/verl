---
name: upgrade-megatron-core
description: Upgrade Megatron-Core in veRL by auditing affected APIs, cross-referencing upstream sources, and updating call sites.
---

# Upgrade Megatron-Core

Upgrade the pinned Megatron-Core version in veRL and update all affected call sites.

## Usage

```
Target Megatron-Core version: $VERSION
```

`$VERSION`: target version tag, e.g. `core_v0.17.0`, `core_r0.12.0`, or a commit SHA.
If omitted, check current pinned version in the Dockerfile and validate compatibility.

## Prerequisites — Clone Upstream Sources

veRL depends on both **Megatron-LM** and **mbridge** (the HF↔MCore weight conversion
layer). Both must be audited when upgrading.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
MCORE_DIR="${REPO_ROOT}/Megatron-LM"
MBRIDGE_DIR="${REPO_ROOT}/mbridge-src"

# Validate VERSION
if [[ ! "$VERSION" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
  echo "Error: Invalid version format"; exit 1
fi

# Clone Megatron-LM
if [ ! -d "$MCORE_DIR" ]; then
  git clone --depth 1 --branch "${VERSION}" https://github.com/NVIDIA/Megatron-LM.git "$MCORE_DIR"
else
  cd "$MCORE_DIR" && git fetch origin && git checkout "${VERSION}" && cd -
fi

# Clone mbridge (check current version from docker/Dockerfile.stable.vllm)
MBRIDGE_VERSION=$(grep "megatron.bridge\|mbridge" docker/Dockerfile.stable.vllm | grep -oP 'v[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [ ! -d "$MBRIDGE_DIR" ]; then
  git clone --depth 1 https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "$MBRIDGE_DIR"
fi
```

If cloning fails, stop and report to the user.

**Relevant upstream source paths:**

```
Megatron-LM/megatron/core/
  parallel_state.py
  distributed/
  optimizer/
  optimizer_param_scheduler.py
  pipeline_parallel/schedules.py
  pipeline_parallel/utils.py
  transformer/transformer_config.py
  transformer/transformer_layer.py
  transformer/enums.py
  transformer/moe/moe_utils.py
  transformer/moe/token_dispatcher.py
  transformer/moe/router.py
  transformer/multi_latent_attention.py
  transformer/multi_token_prediction.py
  models/gpt/gpt_model.py
  models/gpt/gpt_layer_specs.py
  models/gpt/moe_module_specs.py
  models/common/model_chunk_schedule_plan.py
  dist_checkpointing/serialization.py
  dist_checkpointing/strategies/fully_parallel.py
  dist_checkpointing/strategies/async_utils.py
  dist_checkpointing/strategies/filesystem_async.py
  packed_seq_params.py
  utils.py
  inference/contexts.py
  fusions/fused_cross_entropy.py
  extensions/transformer_engine.py
```

---

## Affected Files in veRL

### Primary — Core Model Layer

| File | Key Megatron APIs |
|------|------------------|
| `verl/models/mcore/patch.py` | `parallel_state`, `tensor_parallel`, `multi_latent_attention`, `dist_checkpointing.strategies.*`, monkey-patches megatron internals |
| `verl/models/mcore/model_initializer.py` | `GPTModel`, `get_gpt_decoder_block_spec`, `get_gpt_mtp_block_spec`, `TEColumnParallelLinear`, `TERowParallelLinear`, `MLPSubmodules`, `get_vit_layer_with_transformer_engine_spec` |
| `verl/models/mcore/model_forward_fused.py` | `parallel_state`, `GPTModel`, `PackedSeqParams`, `gather_from_sequence_parallel_region`, `make_viewless_tensor`, `deprecate_inference_params`, `BaseInferenceContext`, `config_logger` |
| `verl/models/mcore/model_forward_1f1b_overlap.py` | `TransformerModelChunkSchedulePlan`, `GPTModel`, `make_viewless_tensor`, `gather_from_sequence_parallel_region`, `MultiTokenPredictionLayer` |
| `verl/models/mcore/mtp_patch.py` | `GPTModel`, `MultiTokenPredictionLayer`, `roll_tensor`, `make_viewless_tensor`, `unwrap_model`, `parallel_state` |
| `verl/models/mcore/config_converter.py` | `parallel_state`, `TransformerConfig`, `MLATransformerConfig`, `set_experimental_flag`, `AttnBackend` |
| `verl/models/mcore/bridge.py` | `parallel_state`, `tensor_parallel`, `megatron.bridge.AutoBridge`, `AutoMapping`, `CanonicalLoRA`, `DoRA`, `LoRA`, `VLMLoRA` |
| `verl/models/mcore/loader.py` | `mpu`, `DistributedDataParallel`, `Float16Module` |

### Primary — Workers

| File | Key Megatron APIs |
|------|------------------|
| `verl/workers/megatron_workers.py` | `parallel_state`, `tensor_parallel`, `AttnBackend`, `dist_checkpointing.strategies.base.async_calls` |
| `verl/utils/megatron_utils.py` | Broad megatron utils; `LayerType`, `finalize_model_grads` |

### Primary — Utilities

| File | Key Megatron APIs |
|------|------------------|
| `verl/utils/megatron/dist_checkpointing.py` | `dist_checkpointing`, `mpu`, `ShardedObjectHistoryRecorder`, `FullyParallelSaveStrategy`, `FullyParallelLoadStrategy` |
| `verl/utils/megatron/optimizer.py` | `OptimizerConfig`, `get_megatron_optimizer`, `OptimizerParamScheduler` |
| `verl/utils/megatron/pipeline_parallel.py` | `parallel_state` |
| `verl/utils/megatron/tensor_parallel.py` | `parallel_state`, `tensor_parallel`, `ModelParallelConfig` |
| `verl/utils/megatron/router_replay_patch.py` | `moe_utils`, `MoEAlltoAllTokenDispatcher`, `TopKRouter`, `TransformerConfig` |
| `verl/utils/megatron/router_replay_utils.py` | `parallel_state`, `get_schedule_table`, `gather/scatter_to_sequence_parallel_region`, `TransformerConfig`, `get_transformer_layer_offset`, `is_vp_first_stage`, `is_vp_last_stage`, `LayerType` |
| `verl/utils/checkpoint/megatron_checkpoint_manager.py` | `dist_checkpointing`, `mpu`, `tensor_parallel`, `ShardedObject`, `AttnBackend`, `megatron.bridge.training.checkpointing`, `async_calls` |
| `verl/trainer/distillation/megatron/losses.py` | `fused_cross_entropy`, `get_tensor_model_parallel_group`, `VocabUtility` |

### Secondary — Model Merger

| File | Key Megatron APIs |
|------|------------------|
| `verl/model_merger/megatron_model_merger.py` | `mpu`, `ModelType`, `model_parallel_cuda_manual_seed` |

### Version / Docker

| File | Usage |
|------|-------|
| `docker/Dockerfile.stable.vllm` | `pip install git+...Megatron-LM.git@core_v0.X.Y` |
| `docker/verl*/Dockerfile.app.*` | May also pin Megatron version |

---

## Audit Workflow

### Step 1: Identify Current Pinned Version

```bash
grep -i "Megatron-LM\|megatron" docker/Dockerfile.stable.vllm | grep "pip install"
```

### Step 2: Enumerate All Megatron Imports

```bash
grep -rn "from megatron\|import megatron" verl/ --include="*.py" | grep -v "__pycache__"
```

### Step 3: Cross-Reference Each Symbol

For each import, verify in `Megatron-LM/`:
- Does the module path still exist?
- Has the class/function signature changed?
- Has anything been moved or renamed?

**Highest-churn symbols to always check:**

| Symbol | Why It Changes Often |
|--------|----------------------|
| `TransformerConfig` | New fields added nearly every release |
| `MLATransformerConfig` | MLA (Multi-Latent Attention) is experimental |
| `PackedSeqParams` | Sequence packing format evolves |
| `get_gpt_decoder_block_spec` / `get_gpt_mtp_block_spec` | Layer spec signatures change |
| `FullyParallelSaveStrategy` / `FullyParallelLoadStrategy` | Distributed checkpointing API evolves |
| `async_calls` | Async checkpoint internals are fragile |
| `MoEAlltoAllTokenDispatcher` | MoE dispatcher is frequently refactored |
| `get_schedule_table` | PP schedule API changes with new schedules |
| `AttnBackend` | New backends added, old ones deprecated |

### Step 4: Audit patch.py Carefully

`verl/models/mcore/patch.py` monkey-patches Megatron internals. These break silently:

```bash
# See what is being patched
grep -n "= " verl/models/mcore/patch.py | grep -v "^#" | head -30
```

For each patched attribute:
- Verify the patched object still exists in `Megatron-LM/`
- Verify its internal structure hasn't changed
- If the upstream bug is fixed at `$VERSION`, consider removing the patch

### Step 5: Audit MoE Router Replay Patches

`router_replay_patch.py` and `router_replay_utils.py` patch MoE token dispatching
internals that change frequently:

```bash
grep -n "from megatron" verl/utils/megatron/router_replay_patch.py
grep -n "from megatron" verl/utils/megatron/router_replay_utils.py
```

Verify `MoEAlltoAllTokenDispatcher`, `TopKRouter`, and `get_schedule_table` signatures.

### Step 6: Audit mbridge (megatron.bridge)

`verl/models/mcore/bridge.py` uses mbridge for HF↔MCore weight conversion. mbridge
version must be compatible with the new Megatron-Core version:

```bash
grep -n "megatron.bridge" verl/models/mcore/bridge.py
grep -n "megatron.bridge" verl/utils/checkpoint/megatron_checkpoint_manager.py
```

Check `mbridge-src/` for:
- `AutoBridge`, `AutoMapping` signatures
- `CanonicalLoRA`, `LoRA`, `VLMLoRA`, `DoRA` interfaces
- `apply_peft_adapter_filter_to_state_dict` in `training/checkpointing`

### Step 7: Update Version Pin

```bash
# Update Dockerfile
sed -i "s|Megatron-LM.git@core_v[0-9.]*|Megatron-LM.git@${VERSION}|" docker/Dockerfile.stable.vllm

# Check other Dockerfiles
grep -rl "Megatron-LM" docker/ | xargs grep -l "pip install"
```

### Step 8: Run Validation

```bash
# CPU import checks
python -c "
from verl.models.mcore import config_converter
from verl.utils.megatron import optimizer, pipeline_parallel
print('Megatron imports OK')
"

# CPU unit tests
pytest tests/test_*_on_cpu.py -x

# If GPU available, run megatron-specific tests
pytest tests/ -k "megatron" -x
```

---

## Common Breaking Patterns

| Pattern | What to Check |
|---------|---------------|
| `TransformerConfig` new required fields | Config constructor calls in `config_converter.py` |
| `PackedSeqParams` field renames | All usages in `model_forward_fused.py` |
| Checkpointing strategy API | `FullyParallelSaveStrategy/LoadStrategy` constructor args |
| `get_gpt_decoder_block_spec` signature | `model_initializer.py` call site |
| MoE dispatcher refactor | `router_replay_patch.py` internal hooks |
| `parallel_state` function renames | All `mpu.*` usages across workers |
| `AttnBackend` enum values | `config_converter.py`, `megatron_workers.py` |
| MTP API changes | `mtp_patch.py` — `MultiTokenPredictionLayer`, `roll_tensor` |
| `async_calls` internals | `megatron_checkpoint_manager.py` async checkpoint path |

---

## Output

Report:

1. Current pinned version vs target version
2. Table of changed/broken imports with fix applied
3. Patches in `patch.py` / `router_replay_patch.py` that need updating or removal
4. mbridge compatibility status
5. Files modified
6. Validation commands run and their results

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================
Location: .agents/skills/upgrade-megatron-core/SKILL.md

## How to Update
- When new megatron imports added to verl: update Affected Files tables
- When new common breaking patterns emerge: update the patterns table
- When mbridge repo URL changes: update Prerequisites section
================================================================================
-->
