# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Integration test for legacy MegatronPPOActor loss normalization.

Verifies that the legacy Megatron worker produces correct loss normalization
that matches the engine path, specifically:

  1) Accumulated gradients are element-wise identical regardless of micro-batch
     splitting (gradient accumulation invariance)
  2) global_batch_info is populated with correct dp_size and batch_num_tokens
     (including correct all_reduce group for CP>1)
  3) global_batch_info is consistent across different micro_batch_size values

Uses an in-memory tiny LlamaForCausalLM — no model files on disk required.

Requirements:
    - Megatron-Core with GPU access
    - Number of GPUs >= TP * PP * CP  (default: TP=1, PP=2 -> 2 GPUs)

Usage (single-node):
    # PP=2, TP=1, CP=1 (2 GPUs) -- default
    torchrun --nproc_per_node=2 \
        tests/workers/actor/test_legacy_megatron_loss_normalization.py

    # PP=2, CP=2 (4 GPUs) -- validates batch_num_tokens with CP>1
    ACTOR_CP=2 torchrun --nproc_per_node=4 \
        tests/workers/actor/test_legacy_megatron_loss_normalization.py
"""

import os
import sys

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from tensordict import TensorDict
from transformers import LlamaConfig

from verl import DataProto
from verl.models.mcore.registry import hf_to_mcore_config
from verl.utils.device import get_device_id
from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config
from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.config import McoreActorConfig

# ---- Configurable via env vars ----
ACTOR_TP = int(os.environ.get("ACTOR_TP", 1))
ACTOR_PP = int(os.environ.get("ACTOR_PP", 2))
ACTOR_CP = int(os.environ.get("ACTOR_CP", 1))
MINI_BATCH_SIZE = int(os.environ.get("MINI_BATCH_SIZE", 4))
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", 16))
RESPONSE_LEN = int(os.environ.get("RESPONSE_LEN", 16))
SEED = 42

# Tiny model hyperparameters (fits in any GPU)
VOCAB_SIZE = 256
HIDDEN = 64
NUM_LAYERS = 4
NUM_HEADS = 4
NUM_KV_HEADS = 2


def log(msg):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"[TEST] {msg}", flush=True)


def make_tiny_hf_config():
    """Create a minimal LlamaForCausalLM config in memory."""
    return LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        architectures=["LlamaForCausalLM"],
    )


def make_synthetic_data(batch_size, prompt_len, response_len, vocab_size, device, variable_lengths=False):
    """Generate deterministic synthetic PPO data."""
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    total_len = prompt_len + response_len

    input_ids = torch.randint(0, vocab_size, (batch_size, total_len), generator=gen).to(device)
    attention_mask = torch.ones(batch_size, total_len, device=device)
    position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(batch_size, -1).clone()
    responses = input_ids[:, -response_len:].clone()

    if variable_lengths:
        response_mask = torch.zeros(batch_size, response_len, device=device)
        for i in range(batch_size):
            valid_len = max(1, response_len - i * (response_len // (batch_size + 1)))
            response_mask[i, :valid_len] = 1.0
    else:
        response_mask = torch.ones(batch_size, response_len, device=device)

    old_log_probs = torch.randn(batch_size, response_len, generator=gen).to(device) * 0.1
    advantages = torch.randn(batch_size, response_len, generator=gen).to(device) * 0.5

    td = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
        },
        batch_size=[batch_size],
    )
    return DataProto(batch=td, meta_info={"temperature": 1.0})


def snapshot_megatron_grads(actor_module):
    """Capture accumulated gradients from Megatron model chunks.

    Megatron uses main_grad for mixed precision; falls back to .grad.
    Only parameters with actual gradients on this PP stage are captured.
    """
    grads = {}
    for chunk_idx, chunk in enumerate(actor_module):
        for name, p in chunk.named_parameters():
            g = getattr(p, "main_grad", None)
            if g is None:
                g = p.grad
            if g is not None:
                grads[f"chunk{chunk_idx}/{name}"] = g.detach().clone()
    return grads


def run_update_with_grads(actor, actor_module, optimizer, data, micro_batch_size, saved_state):
    """Reset model, run update_policy, return (grad_norm, grads_snapshot, global_batch_info)."""
    for chunk_new, chunk_saved in zip(actor_module, saved_state, strict=True):
        for p_new, p_saved in zip(chunk_new.parameters(), chunk_saved, strict=True):
            p_new.data.copy_(p_saved.data)

    optimizer.zero_grad()
    for chunk in actor_module:
        chunk.zero_grad_buffer()

    data_with_micro_bs = DataProto(
        batch=data.batch.clone(),
        meta_info={**data.meta_info, "micro_batch_size": micro_batch_size},
    )

    captured_grads = {}
    module_ref = actor.actor_module

    orig_step = actor.actor_optimizer.step

    def _patched_step():
        captured_grads.update(snapshot_megatron_grads(module_ref))
        return orig_step()

    actor.actor_optimizer.step = _patched_step
    try:
        metrics = actor.update_policy(dataloader=[data_with_micro_bs])
    finally:
        actor.actor_optimizer.step = orig_step

    grad_norm = metrics.get("actor/grad_norm", [None])
    if isinstance(grad_norm, list):
        grad_norm = grad_norm[0]

    gbi = dict(actor.config.global_batch_info)
    return grad_norm, captured_grads, gbi


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="cpu:gloo,cuda:nccl", timeout=dist.default_pg_timeout)

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=ACTOR_TP,
        pipeline_model_parallel_size=ACTOR_PP,
        context_parallel_size=ACTOR_CP,
    )

    dp_size = mpu.get_data_parallel_world_size()
    log(f"Parallel config: TP={ACTOR_TP}, PP={ACTOR_PP}, CP={ACTOR_CP}, DP={dp_size}, world={world_size}")

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(SEED)

    # ---- Build model from in-memory config ----
    hf_config = make_tiny_hf_config()
    tf_config = hf_to_mcore_config(hf_config, torch.bfloat16)

    wrap_config = McoreModuleWrapperConfig(
        is_value_model=False,
        share_embeddings_and_output_weights=False,
        wrap_with_ddp=True,
        use_distributed_optimizer=True,
    )

    actor_module, tf_config = make_megatron_module(
        wrap_config=wrap_config,
        tf_config=tf_config,
        hf_config=hf_config,
    )
    log(f"Tiny model built: {NUM_LAYERS} layers, hidden={HIDDEN}, {len(actor_module)} chunks")

    # ---- Build optimizer ----
    class _OptimCfg:
        """Minimal config satisfying both attribute access and .get()."""

        optimizer = "adam"
        lr = 1e-4
        min_lr = 0.0
        weight_decay = 0.0
        clip_grad = 999.0

        def get(self, key, default=None):
            return getattr(self, key, default)

    optim_config = init_megatron_optim_config(_OptimCfg())
    optimizer = get_megatron_optimizer(actor_module, optim_config)

    # ---- Build actor ----
    actor_config = McoreActorConfig(
        ppo_mini_batch_size=MINI_BATCH_SIZE,
        ppo_micro_batch_size_per_gpu=MINI_BATCH_SIZE,
        ppo_epochs=1,
        clip_ratio=0.2,
        entropy_coeff=0.0,
        loss_agg_mode="token-mean",
        use_dynamic_bsz=False,
        rollout_n=1,
    )

    actor = MegatronPPOActor(
        config=actor_config,
        model_config=None,
        hf_config=hf_config,
        tf_config=tf_config,
        actor_module=actor_module,
        actor_optimizer=optimizer,
    )
    log("MegatronPPOActor created")

    # ---- Save initial state ----
    saved_state = []
    for chunk in actor_module:
        saved_state.append([p.data.clone() for p in chunk.parameters()])

    device = get_device_id()
    micro_batch_sizes = [MINI_BATCH_SIZE]
    if MINI_BATCH_SIZE >= 2:
        micro_batch_sizes.append(MINI_BATCH_SIZE // 2)
    if MINI_BATCH_SIZE >= 4:
        micro_batch_sizes.append(MINI_BATCH_SIZE // 4)

    all_ok = True

    for variable_lengths in [False, True]:
        mask_label = "variable mask" if variable_lengths else "uniform mask"
        log(f"\n{'=' * 60}")
        log(f"  Test: {mask_label}")
        log(f"{'=' * 60}")

        data = make_synthetic_data(
            MINI_BATCH_SIZE,
            PROMPT_LEN,
            RESPONSE_LEN,
            VOCAB_SIZE,
            device,
            variable_lengths=variable_lengths,
        )
        local_valid_tokens = int(data.batch["response_mask"].sum().item())
        log(
            f"  Synthetic data: batch_size={MINI_BATCH_SIZE}, prompt={PROMPT_LEN}, "
            f"response={RESPONSE_LEN}, local_valid_tokens={local_valid_tokens}"
        )

        results = {}
        grads_by_mbs = {}
        gbi_by_mbs = {}
        for mbs in micro_batch_sizes:
            n_micro = MINI_BATCH_SIZE // mbs
            gn, grads, gbi = run_update_with_grads(
                actor,
                actor_module,
                optimizer,
                data,
                mbs,
                saved_state,
            )
            results[mbs] = gn
            grads_by_mbs[mbs] = grads
            gbi_by_mbs[mbs] = gbi
            log(f"    micro_batch_size={mbs} (N={n_micro}): grad_norm={gn}, captured {len(grads)} grad tensors")

        gbi_first = gbi_by_mbs[micro_batch_sizes[0]]

        # ---- A. Validate global_batch_info ----
        dp_size_with_cp = mpu.get_data_parallel_world_size(with_context_parallel=True)
        dp_size_pure = mpu.get_data_parallel_world_size()

        log(f"\n  --- global_batch_info validation [{mask_label}] (CP={ACTOR_CP}) ---")

        if gbi_first is None or "dp_size" not in gbi_first:
            log("  FAIL: global_batch_info not populated")
            all_ok = False
        else:
            actual_dp = gbi_first["dp_size"]
            actual_bnt = gbi_first["batch_num_tokens"]
            expected_bnt = local_valid_tokens * dp_size_with_cp

            log(f"    global_batch_info.dp_size = {actual_dp}  (expected {dp_size_pure})")
            log(f"    global_batch_info.batch_num_tokens = {actual_bnt}  (expected {expected_bnt})")

            if actual_dp != dp_size_pure:
                log(f"    FAIL: dp_size should be pure DP={dp_size_pure}, got {actual_dp}")
                all_ok = False
            else:
                log(f"    OK: dp_size = {dp_size_pure} (pure DP, without CP)")

            if actual_bnt != expected_bnt:
                log(
                    f"    FAIL: batch_num_tokens should be {expected_bnt} "
                    f"(= {local_valid_tokens} local x {dp_size_with_cp} DP*CP ranks), got {actual_bnt}"
                )
                all_ok = False
            else:
                log(
                    f"    OK: batch_num_tokens = {expected_bnt} "
                    f"(= {local_valid_tokens} local x {dp_size_with_cp} DP*CP ranks)"
                )

            if ACTOR_CP > 1:
                wrong_bnt = local_valid_tokens * dp_size_pure
                log(f"    (If all_reduce were DP-only, batch_num_tokens would be {wrong_bnt} -- {ACTOR_CP}x too small)")

        # ---- global_batch_info must be identical across micro_batch_sizes ----
        if len(gbi_by_mbs) > 1:
            log(f"\n  --- global_batch_info consistency [{mask_label}] ---")
            ref_mbs_gbi = micro_batch_sizes[0]
            for mbs in micro_batch_sizes[1:]:
                if gbi_by_mbs[mbs] != gbi_by_mbs[ref_mbs_gbi]:
                    log(
                        f"    FAIL: gbi differs for micro_bs={ref_mbs_gbi} vs {mbs}: "
                        f"{gbi_by_mbs[ref_mbs_gbi]} vs {gbi_by_mbs[mbs]}"
                    )
                    all_ok = False
                else:
                    log(f"    OK: micro_bs={mbs} gbi == micro_bs={ref_mbs_gbi} gbi")

        # ---- B. Check element-wise gradient identity ----
        ref_mbs = MINI_BATCH_SIZE
        ref_grads = grads_by_mbs.get(ref_mbs, {})

        if not ref_grads:
            log("  WARNING: no gradients captured (expected for non-last PP stage)")
            log(f"  SKIPPED gradient check [{mask_label}]")
            continue

        for mbs in micro_batch_sizes:
            if mbs == ref_mbs:
                continue
            test_grads = grads_by_mbs[mbs]
            n_micro = MINI_BATCH_SIZE // mbs
            log(f"\n  --- gradient comparison [{mask_label}]: micro_bs={ref_mbs} vs micro_bs={mbs} (N={n_micro}) ---")

            if set(ref_grads.keys()) != set(test_grads.keys()):
                log("    FAIL: different parameter sets")
                all_ok = False
                continue

            for name in sorted(ref_grads.keys()):
                g_ref = ref_grads[name]
                g_test = test_grads[name]
                max_diff = (g_ref - g_test).abs().max().item()
                scale = g_ref.abs().max().item() + 1e-12
                rel_err = max_diff / scale
                ok = rel_err < 1e-2
                status = "OK" if ok else "FAIL"
                log(f"    {name}: max_abs_diff={max_diff:.2e}, rel_err={rel_err:.2e} [{status}]")
                if not ok:
                    all_ok = False

        # ---- C. Check grad_norm invariant ----
        ref_gn = results[ref_mbs]
        if ref_gn is not None and ref_gn != 0:
            log(
                f"\n  --- grad_norm invariance [{mask_label}] "
                f"(reference: micro_bs={ref_mbs}, grad_norm={ref_gn:.6f}) ---"
            )
            for mbs, gn in sorted(results.items(), reverse=True):
                if gn is None or gn == 0:
                    continue
                ratio = gn / ref_gn
                ok = abs(ratio - 1.0) < 0.01
                status = "OK" if ok else "FAIL"
                n_micro = MINI_BATCH_SIZE // mbs
                log(f"    micro_bs={mbs} (N={n_micro}): grad_norm={gn:.6f}, ratio={ratio:.6f} [{status}]")
                if not ok:
                    all_ok = False

    log("")
    if all_ok:
        log("PASSED: all checks OK")
    else:
        log("FAILED: see errors above")
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
