# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 tests/utils/test_special_megatron_kl_loss_tp.py"""

import gc
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

import megatron.core.parallel_state as mpu

from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk as compute_forward_kl_topk_ref
from verl.trainer.distillation.megatron.losses import compute_forward_kl_topk as compute_forward_kl_topk_vp
from verl.workers.config import DistillationConfig
from verl.utils.distributed import initialize_global_process_group, destroy_global_process_group

MAX_TEST_CASES = int(os.environ.get("MAX_TEST_CASES", 4))


class TestVocabParallelKLDivergence:
    def __init__(self):
        local_rank, rank, world_size = initialize_global_process_group()
        mpu.initialize_model_parallel(tensor_model_parallel_size=world_size)

        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.group = mpu.get_tensor_model_parallel_group()
        print(f"[INFO]: Local rank: {self.local_rank}, World size: {self.world_size}")

    def initialize(self, test_case_idx: int):
        self.test_case_idx = test_case_idx

    def shutdown(self):
        destroy_global_process_group()

    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.synchronize()

    def generate_hyper(self):
        if self.test_case_idx == 0:
            self.batch_size = 2
            self.seq_len = 4
            self.vocab_size = 32
            self.topk = 6
            self.clamp = -6.0
        elif self.test_case_idx == 1:
            self.batch_size = 4
            self.seq_len = 8
            self.vocab_size = 64
            self.topk = 8
            self.clamp = -8.0
        elif self.test_case_idx == 2:
            self.batch_size = 2
            self.seq_len = 4
            self.vocab_size = 128
            self.topk = 10
            self.clamp = -4.0
        elif self.test_case_idx == 3:
            self.batch_size = 1
            self.seq_len = 16
            self.vocab_size = 64
            self.topk = 4
            self.clamp = None
        else:
            raise ValueError(f"Invalid test case index: {self.test_case_idx}")

        assert self.vocab_size % self.world_size == 0, "vocab_size must be divisible by world_size"
        self.shard_size = self.vocab_size // self.world_size

    def generate_forward_inputs(self):
        B, S, V = self.batch_size, self.seq_len, self.vocab_size
        topk = self.topk
        shard_size = self.shard_size

        full_student_logits = torch.randn(B, S, V, device=self.device) * 0.7
        teacher_full_logits = torch.randn(B, S, V, device=self.device) * 0.9
        teacher_full_logps = F.log_softmax(teacher_full_logits, dim=-1)
        teacher_topk_logps, teacher_topk_indices = torch.topk(teacher_full_logps, k=topk, dim=-1)

        # Edge case 1: Force index 0 collision on rank 1 (on rank 1, global index shard_size maps to local index 0)
        # 1. When a teacher top-k index is not in the local shard, it's remapped to local index 0 (as a dummy placeholder, with its prob set to 0)                                                                                                                                                                                                                                                
        # 2. But local index 0 might also be a legitimate teacher top-k index (e.g., on rank 1, global index shard_size maps to local index 0) 
        teacher_topk_indices[..., 0] = shard_size
        teacher_topk_logps[..., 0] = teacher_full_logps[..., shard_size]

        # Edge case 2: Make the colliding token active (high probability, not clamped)
        full_student_logits[..., shard_size] = 3.0

        # Edge case 3: Force out-of-shard entries for rank 1 (indices 1 and 2 are in rank 0's shard)
        teacher_topk_indices[..., -1] = 1
        teacher_topk_logps[..., -1] = teacher_full_logps[..., 1]
        teacher_topk_indices[..., -2] = 2
        teacher_topk_logps[..., -2] = teacher_full_logps[..., 2]

        # Edge case 4: Force some student probs to be clamped (very low logits)
        full_student_logits.scatter_(
            dim=-1, index=teacher_topk_indices[..., 1:2], src=torch.full((B, S, 1), -50.0, device=self.device)
        )

        return full_student_logits, teacher_topk_logps, teacher_topk_indices

    def verify_correctness(self, iterations: int = 5):
        self.cleanup()
        self.generate_hyper()

        cfg_megatron = DistillationConfig(
            strategy="megatron", rollout_n=-1, ppo_micro_batch_size_per_gpu=-1, log_prob_min_clamp=self.clamp
        )
        config_fsdp = DistillationConfig(
            strategy="fsdp", rollout_n=-1, ppo_micro_batch_size_per_gpu=-1, log_prob_min_clamp=self.clamp
        )

        shard_start = self.local_rank * self.shard_size
        shard_end = shard_start + self.shard_size

        for i in range(iterations):
            if self.local_rank == 0:
                torch.manual_seed(42 + self.test_case_idx * 100 + i)

            # Generate inputs and broadcast to all ranks
            full_student_logits, teacher_topk_logps, teacher_topk_indices = self.generate_forward_inputs()
            dist.broadcast(full_student_logits, src=0, group=self.group)
            dist.broadcast(teacher_topk_logps, src=0, group=self.group)
            dist.broadcast(teacher_topk_indices, src=0, group=self.group)

            # VP implementation on sharded logits
            vp_logits = full_student_logits[..., shard_start:shard_end].contiguous().detach().requires_grad_(True)
            vp_loss, vp_student_mass, vp_teacher_mass = compute_forward_kl_topk_vp(
                student_logits=vp_logits, teacher_topk_log_probs=teacher_topk_logps, teacher_topk_indices=teacher_topk_indices, config=cfg_megatron
            )
            vp_loss.sum().backward()
            grad_vp = vp_logits.grad.detach().clone()

            # Reference implementation on full logits
            full_ref = full_student_logits.detach().clone().requires_grad_(True)
            ref_loss, ref_student_mass, ref_teacher_mass = compute_forward_kl_topk_ref(
                student_logits=full_ref, teacher_topk_log_probs=teacher_topk_logps, teacher_topk_indices=teacher_topk_indices, config=config_fsdp
            )
            ref_loss.sum().backward()
            grad_ref_shard = full_ref.grad[..., shard_start:shard_end].detach().clone()

            # Compare losses
            torch.testing.assert_close(vp_loss, ref_loss, atol=1e-4, rtol=1e-4)

            # Compare gradients
            torch.testing.assert_close(grad_vp, grad_ref_shard, atol=1e-4, rtol=1e-4)

        if self.local_rank == 0:
            print(f"[PASS] VP KL divergence correctness verified for test case {self.test_case_idx}")


if __name__ == "__main__":
    assert int(os.environ.get("WORLD_SIZE", 1)) > 1, (
        "[ERROR]: This test is designed to run in distributed mode with torchrun. "
        "Please use torchrun to execute this script."
    )
    torch.manual_seed(42 + int(os.environ.get("RANK", 0)))

    test = TestVocabParallelKLDivergence()
    for test_case_idx in range(MAX_TEST_CASES):
        if test.local_rank == 0:
            print(f"[INFO] Running test case {test_case_idx}")
        test.initialize(test_case_idx)
        test.verify_correctness()
    test.shutdown()
