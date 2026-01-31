#!/usr/bin/env python3
import os
import sys
import types
import torch
import torch.nn.functional as F
import torch.distributed as dist

from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk as compute_forward_kl_topk_ref
from verl.trainer.distillation.megatron.losses import compute_forward_kl_topk as compute_forward_kl_topk_vp
from verl.workers.config import DistillationConfig

import verl.trainer.distillation.megatron.utils as vp_utils


def setup_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return rank, world, device


def patch_megatron_accessors():
    # Patch only what verl megatron loss imports
    try:
        import megatron.core.parallel_state as ps
    except Exception:
        megatron = sys.modules.get("megatron", types.ModuleType("megatron"))
        core = sys.modules.get("megatron.core", types.ModuleType("megatron.core"))
        ps = types.ModuleType("megatron.core.parallel_state")
        sys.modules["megatron"] = megatron
        sys.modules["megatron.core"] = core
        sys.modules["megatron.core.parallel_state"] = ps

    ps.get_tensor_model_parallel_group = lambda: dist.group.WORLD
    ps.get_tensor_model_parallel_rank = lambda: dist.get_rank()
    ps.get_tensor_model_parallel_world_size = lambda: dist.get_world_size()

    try:
        import megatron.core.tensor_parallel.utils as tpu
    except Exception:
        tensor_parallel = sys.modules.get(
            "megatron.core.tensor_parallel", types.ModuleType("megatron.core.tensor_parallel")
        )
        tpu = types.ModuleType("megatron.core.tensor_parallel.utils")
        sys.modules["megatron.core.tensor_parallel"] = tensor_parallel
        sys.modules["megatron.core.tensor_parallel.utils"] = tpu

    class VocabUtility:
        @staticmethod
        def vocab_range_from_per_partition_vocab_size(partition_vocab_size, rank, world_size):
            start = rank * partition_vocab_size
            end = start + partition_vocab_size
            return start, end

    tpu.VocabUtility = VocabUtility


def patch_vocab_parallel_softmax():
    # Avoid megatron fused ops
    def _softmax(vp_logits: torch.Tensor):
        local_max = vp_logits.max(dim=-1).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        shifted = vp_logits - global_max.unsqueeze(-1)

        exp_logits = torch.exp(shifted)
        local_sum = exp_logits.sum(dim=-1)
        global_sum = local_sum.clone()
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
        return exp_logits, global_sum

    vp_utils.vocab_parallel_softmax = _softmax


def main():
    rank, world, device = setup_dist()
    patch_megatron_accessors()
    patch_vocab_parallel_softmax()

    # Minimal but enough to show the issue
    B, S = 2, 3
    V_total = 32
    assert V_total % world == 0
    V_shard = V_total // world
    topk = 6
    clamp = -6.0

    cfg_vp  = DistillationConfig(strategy="megatron", rollout_n=-1, ppo_micro_batch_size_per_gpu=-1, log_prob_min_clamp=clamp)
    cfg_ref = DistillationConfig(strategy="fsdp",     rollout_n=-1, ppo_micro_batch_size_per_gpu=-1, log_prob_min_clamp=clamp)

    shard_start = rank * V_shard
    shard_end = shard_start + V_shard

    # Try many trials and stop on first mismatch
    N = int(os.environ.get("NTRIALS", "500"))

    for t in range(N):
        # Deterministic per-trial seed on rank0, then broadcast tensors
        if rank == 0:
            torch.manual_seed(1234 + t)

            full_student_logits = torch.randn(B, S, V_total, device=device) * 0.7
            teacher_full_logits = torch.randn(B, S, V_total, device=device) * 0.9
            teacher_full_logps = F.log_softmax(teacher_full_logits, dim=-1)
            teacher_topk_logps, teacher_topk_indices = torch.topk(teacher_full_logps, k=topk, dim=-1)

            # FORCE the collision-at-local-0 on rank1 (global token == V_shard)
            teacher_topk_indices[..., 0] = V_shard
            teacher_topk_logps[..., 0] = teacher_full_logps[..., V_shard]

            # ALSO force a couple out-of-shard entries late in the list so local index 0 repeats AFTER the real one
            teacher_topk_indices[..., -1] = 1
            teacher_topk_logps[..., -1] = teacher_full_logps[..., 1]
            teacher_topk_indices[..., -2] = 2
            teacher_topk_logps[..., -2] = teacher_full_logps[..., 2]

            # Make the colliding real token ACTIVE (not clamped)
            full_student_logits[..., V_shard] = 3.0

            # Make some other teacher topk entries clamped to exercise clamp masking
            full_student_logits.scatter_(
                -1, teacher_topk_indices[..., 1:2], torch.full((B, S, 1), -50.0, device=device)
            )
        else:
            full_student_logits = torch.empty(B, S, V_total, device=device)
            teacher_topk_logps = torch.empty(B, S, topk, device=device)
            teacher_topk_indices = torch.empty(B, S, topk, device=device, dtype=torch.long)

        dist.broadcast(full_student_logits, src=0)
        dist.broadcast(teacher_topk_logps, src=0)
        dist.broadcast(teacher_topk_indices, src=0)

        # VP shard
        vp_logits = full_student_logits[..., shard_start:shard_end].contiguous().detach().requires_grad_(True)
        vp_loss, _, _ = compute_forward_kl_topk_vp(vp_logits, teacher_topk_logps, teacher_topk_indices, cfg_vp)
        vp_loss.sum().backward()
        grad_vp = vp_logits.grad.detach().clone()

        # REF full -> shard
        full_ref = full_student_logits.detach().clone().requires_grad_(True)
        ref_loss, _, _ = compute_forward_kl_topk_ref(full_ref, teacher_topk_logps, teacher_topk_indices, cfg_ref)
        ref_loss.sum().backward()
        grad_ref_shard = full_ref.grad[..., shard_start:shard_end].detach().clone()

        abs_diff = (grad_vp - grad_ref_shard).abs()
        max_diff = abs_diff.max()

        # If mismatch on any rank, flag globally and print the first case
        bad = (max_diff > 1e-5).to(torch.int32)
        dist.all_reduce(bad, op=dist.ReduceOp.SUM)
        if bad.item() > 0:
            if rank == 0:
                print("\n=== FOUND MISMATCH ===")
                print(f"trial={t} max_abs_diff(rank0)={max_diff.item():.6e}")
                print("teacher_topk_indices[0,0]:", teacher_topk_indices[0, 0].tolist())
            # print per-rank summary
            print(f"[rank {rank}] trial={t} max_abs_diff={max_diff.item():.6e}")

            # On rank1, print the colliding token (local 0 == global V_shard)
            if world == 2 and rank == 1:
                with torch.no_grad():
                    student_logp = float(F.log_softmax(full_student_logits, dim=-1)[0, 0, V_shard].item())
                gv = float(grad_vp[0, 0, 0].item())
                gr = float(grad_ref_shard[0, 0, 0].item())
                print(f"[rank 1] local0(global={V_shard}) student_logp={student_logp:.4f} grad_vp={gv:.6e} grad_ref={gr:.6e}")

            dist.destroy_process_group()
            return

    if rank == 0:
        print(f"\nNo mismatch found in {N} trials (note: duplicate-index assignment can be nondeterministic).")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
