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
Manual distributed integration test for legacy FSDP loss normalization.

This is NOT a standard pytest unit test.  It is intended to be launched
via ``torchrun`` (or an equivalent distributed launcher) to verify the
loss-normalization fix in DataParallelPPOActor on real FSDP-wrapped
models with actual gradient reduction.

Checks (using variable-length response_mask -- the case most sensitive
to the mean-of-means bug):

  1) Accumulated gradients are element-wise identical regardless of
     micro-batch splitting (gradient accumulation invariance).
  2) global_batch_info is populated with correct dp_size and
     batch_num_tokens.
  3) global_batch_info is consistent across different micro_batch_size
     values for the same mini-batch.

Uses a tiny in-memory model wrapped with FSDP2 (fully_shard).
No model files on disk required.

Requirements:
    - PyTorch with CUDA and FSDP2 support
    - At least 2 GPUs (or 1 GPU with WORLD_SIZE=1 for smoke-testing)

Usage:
    torchrun --nproc_per_node=2 \
        tests/workers/actor/test_legacy_fsdp_loss_normalization.py

    MINI_BATCH_SIZE=8 torchrun --nproc_per_node=4 \
        tests/workers/actor/test_legacy_fsdp_loss_normalization.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict

from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import FSDPModule, fully_shard
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import FSDPActorConfig, OptimizerConfig

# ---- Configurable via env vars ----
MINI_BATCH_SIZE = int(os.environ.get("MINI_BATCH_SIZE", 4))
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", 8))
RESPONSE_LEN = int(os.environ.get("RESPONSE_LEN", 8))
SEED = 42

VOCAB_SIZE = 256
HIDDEN = 64


def log(msg):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"[TEST] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Tiny model
# ---------------------------------------------------------------------------


class TinyBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.proj(x)


class TinyModel(nn.Module):
    """Deterministic tiny LM for reproducible gradient checks."""

    _no_split_modules = ["TinyBlock"]

    def __init__(self, vocab_size=VOCAB_SIZE, hidden=HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.block = TinyBlock(hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kw):
        h = self.block(self.embed(input_ids))
        logits = self.head(h)

        class _Out:
            pass

        o = _Out()
        o.logits = logits
        return o


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fsdp_model(device, seed=SEED):
    """Create a TinyModel wrapped with FSDP2 (fully_shard)."""
    torch.manual_seed(seed)
    model = TinyModel().to(device)
    fully_shard(model.block)
    fully_shard(model)
    assert isinstance(model, FSDPModule), f"fully_shard did not produce FSDPModule, got {type(model)}"
    return model


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


def snapshot_fsdp_grads(model):
    """Capture accumulated gradients from an FSDP model before clip/step.

    For DTensor parameters (FSDP2), extracts the local shard's grad so
    that comparisons are apples-to-apples across independent runs.
    """
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if hasattr(g, "to_local"):
            g = g.to_local()
        grads[name] = g.detach().clone()
    return grads


def run_update_with_grads(micro_bs, mini_bs, device, variable_lengths=False, seed=SEED):
    """Build a fresh model+actor, run update_policy, capture raw grads.

    Returns (grad_norm, grads_dict, global_batch_info).
    """
    model = make_fsdp_model(device, seed=seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    config = FSDPActorConfig(
        strategy="fsdp2",
        ppo_mini_batch_size=mini_bs,
        ppo_micro_batch_size_per_gpu=micro_bs,
        ppo_epochs=1,
        clip_ratio=0.2,
        entropy_coeff=0.0,
        grad_clip=999.0,
        use_dynamic_bsz=False,
        use_torch_compile=False,
        ulysses_sequence_parallel_size=1,
        optim=OptimizerConfig(lr=1.0),
        rollout_n=1,
    )

    actor = DataParallelPPOActor(config=config, actor_module=model, actor_optimizer=optimizer)

    captured_grads = {}

    orig_optimizer_step = actor._optimizer_step

    def _patched_optimizer_step():
        captured_grads.update(snapshot_fsdp_grads(model))
        return orig_optimizer_step()

    actor._optimizer_step = _patched_optimizer_step

    data = make_synthetic_data(mini_bs, PROMPT_LEN, RESPONSE_LEN, VOCAB_SIZE, device, variable_lengths=variable_lengths)
    metrics = actor.update_policy(data)

    grad_norm = metrics.get("actor/grad_norm")
    if isinstance(grad_norm, list):
        grad_norm = grad_norm[0]

    gbi = dict(actor.config.global_batch_info)
    return grad_norm, captured_grads, gbi


# ---------------------------------------------------------------------------
# Main — run via torchrun, not pytest
# ---------------------------------------------------------------------------


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not torch.cuda.is_available():
        print("SKIP: this test requires CUDA", file=sys.stderr)
        sys.exit(0)
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="cpu:gloo,cuda:nccl", init_method="env://")

    log(f"FSDP loss normalization test: world_size={world_size}")

    device = get_device_id()

    micro_batch_sizes = [MINI_BATCH_SIZE]
    if MINI_BATCH_SIZE >= 2:
        micro_batch_sizes.append(MINI_BATCH_SIZE // 2)
    if MINI_BATCH_SIZE >= 4:
        micro_batch_sizes.append(MINI_BATCH_SIZE // 4)

    all_ok = True

    # Variable-length response_mask only — the case most sensitive to the
    # mean-of-means bug.  Uniform-mask baseline is covered by the CPU unit
    # tests in test_agg_loss_math.py.
    log(f"\n{'=' * 60}")
    log("  Test: variable-length response mask")
    log(f"{'=' * 60}")

    data_for_info = make_synthetic_data(
        MINI_BATCH_SIZE, PROMPT_LEN, RESPONSE_LEN, VOCAB_SIZE, device, variable_lengths=True
    )
    local_valid_tokens = int(data_for_info.batch["response_mask"].sum().item())

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
            micro_bs=mbs, mini_bs=MINI_BATCH_SIZE, device=device, variable_lengths=True
        )
        results[mbs] = gn
        grads_by_mbs[mbs] = grads
        gbi_by_mbs[mbs] = gbi
        log(f"    micro_batch_size={mbs} (N={n_micro}): grad_norm={gn}, captured {len(grads)} grad tensors")

    gbi_first = gbi_by_mbs[micro_batch_sizes[0]]

    # ---- A. Validate global_batch_info ----
    log("\n  --- global_batch_info validation ---")

    if gbi_first is None or "dp_size" not in gbi_first or "batch_num_tokens" not in gbi_first:
        log("  FAIL: global_batch_info not populated (missing dp_size and/or batch_num_tokens)")
        all_ok = False
    else:
        actual_dp = gbi_first["dp_size"]
        actual_bnt = gbi_first["batch_num_tokens"]
        expected_bnt = local_valid_tokens * world_size

        log(f"    global_batch_info.dp_size = {actual_dp}  (expected {float(world_size)})")
        log(f"    global_batch_info.batch_num_tokens = {actual_bnt}  (expected {expected_bnt})")

        if actual_dp != float(world_size):
            log(f"    FAIL: dp_size should be {float(world_size)}, got {actual_dp}")
            all_ok = False
        else:
            log(f"    OK: dp_size = {actual_dp}")

        if actual_bnt != expected_bnt:
            log(
                f"    FAIL: batch_num_tokens should be {expected_bnt} "
                f"(= {local_valid_tokens} local x {world_size} ranks), got {actual_bnt}"
            )
            all_ok = False
        else:
            log(f"    OK: batch_num_tokens = {expected_bnt} (= {local_valid_tokens} local x {world_size} ranks)")

    # ---- global_batch_info must be identical across micro_batch_sizes ----
    if len(gbi_by_mbs) > 1:
        log("\n  --- global_batch_info consistency ---")
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
        log("  WARNING: no gradients captured — skipping gradient check")
    else:
        for mbs in micro_batch_sizes:
            if mbs == ref_mbs:
                continue
            test_grads = grads_by_mbs[mbs]
            n_micro = MINI_BATCH_SIZE // mbs
            log(f"\n  --- gradient comparison: micro_bs={ref_mbs} vs micro_bs={mbs} (N={n_micro}) ---")

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
                ok = rel_err < 5e-3
                status = "OK" if ok else "FAIL"
                log(f"    {name}: max_abs_diff={max_diff:.2e}, rel_err={rel_err:.2e} [{status}]")
                if not ok:
                    all_ok = False

    # ---- C. Check grad_norm invariance ----
    ref_gn = results[ref_mbs]
    if ref_gn is not None and ref_gn != 0:
        log(f"\n  --- grad_norm invariance (reference: micro_bs={ref_mbs}, grad_norm={ref_gn:.6f}) ---")
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
