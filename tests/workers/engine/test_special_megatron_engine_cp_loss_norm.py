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
"""
Megatron engine path: batch_num_tokens correctness with CP > 1.

Verifies that MegatronEngine.forward_backward_batch all-reduces
batch_num_tokens over the DP×CP group (with_context_parallel=True),
not just the pure DP group.  When CP > 1 each CP rank holds only 1/CP
of the sequence, so using the pure DP group would undercount tokens by
a factor of CP, inflating gradients by the same factor.

Usage::

    torchrun --standalone --nnodes=1 --nproc-per-node=8 \
        tests/workers/engine/test_special_megatron_engine_cp_loss_norm.py
"""

import os

import megatron.core.parallel_state as mpu
import torch
import torch.distributed as dist

from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

ACTOR_TP = int(os.environ.get("ACTOR_TP", 1))
ACTOR_PP = int(os.environ.get("ACTOR_PP", 2))
ACTOR_CP = int(os.environ.get("ACTOR_CP", 2))

MINI_BATCH_SIZE = int(os.environ.get("MINI_BATCH_SIZE", 4))
RESPONSE_LEN = int(os.environ.get("RESPONSE_LEN", 16))
SEED = 42


class TestBatchNumTokensCPGroup:
    """batch_num_tokens must be all-reduced over DP×CP for correct loss normalization."""

    def __init__(self):
        self.local_rank, self.rank, self.world_size = initialize_global_process_group()
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=ACTOR_TP,
            pipeline_model_parallel_size=ACTOR_PP,
            context_parallel_size=ACTOR_CP,
        )
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(SEED)

        self.dp_size = mpu.get_data_parallel_world_size()
        self.cp_size = mpu.get_context_parallel_world_size()
        self.dp_cp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        self.device = torch.device(f"cuda:{self.local_rank}")

        if self.rank == 0:
            print(
                f"[INFO] Parallel config: TP={ACTOR_TP} PP={ACTOR_PP} CP={ACTOR_CP} "
                f"DP={self.dp_size} world={self.world_size}"
            )

    def shutdown(self):
        destroy_global_process_group()

    # ------------------------------------------------------------------
    # Test 1: DP×CP all-reduce gives correct global token count
    # ------------------------------------------------------------------
    def test_dp_cp_allreduce_gives_global_token_count(self):
        """All-reducing local token counts over DP×CP must yield the true
        global total.  This is the group that forward_backward_batch must use."""

        if self.rank == 0:
            print("[TEST 1] DP×CP all-reduce gives correct global token count")

        tokens_per_cp_rank = MINI_BATCH_SIZE * RESPONSE_LEN // self.cp_size
        local_tokens = torch.tensor(tokens_per_cp_rank, dtype=torch.float32, device=self.device)

        expected_global = float(self.dp_size * self.cp_size * tokens_per_cp_rank)

        bnt = local_tokens.clone()
        dist.all_reduce(
            bnt,
            op=dist.ReduceOp.SUM,
            group=mpu.get_data_parallel_group(with_context_parallel=True),
        )

        if self.rank == 0:
            print(f"  local tokens/rank    = {tokens_per_cp_rank}")
            print(f"  DP×CP all-reduce     = {bnt.item()}")
            print(f"  expected global total = {expected_global}")

        assert bnt.item() == expected_global, f"DP×CP all-reduce gave {bnt.item()}, expected {expected_global}"

        # Sanity: pure DP group gives a different (smaller) result when CP > 1
        if self.cp_size > 1:
            bnt_dp_only = local_tokens.clone()
            dist.all_reduce(
                bnt_dp_only,
                op=dist.ReduceOp.SUM,
                group=mpu.get_data_parallel_group(),
            )
            assert bnt_dp_only.item() < expected_global, (
                f"Sanity check: pure DP all-reduce should be smaller than global "
                f"total when CP > 1, got {bnt_dp_only.item()} vs {expected_global}"
            )
            if self.rank == 0:
                print(f"  (sanity: pure DP all-reduce = {bnt_dp_only.item()}, which is {self.cp_size}× too small)")

        if self.rank == 0:
            print("[PASS] test_dp_cp_allreduce_gives_global_token_count\n")

    # ------------------------------------------------------------------
    # Test 2: agg_loss is correct with DP×CP batch_num_tokens
    # ------------------------------------------------------------------
    def test_agg_loss_with_correct_bnt(self):
        """agg_loss with DP×CP batch_num_tokens must produce the correct
        token-mean loss (= local_sum / global_tokens * dp_size)."""

        if self.rank == 0:
            print("[TEST 2] agg_loss correctness with DP×CP batch_num_tokens")

        from verl.trainer.ppo.core_algos import agg_loss

        tokens_per_cp_rank = MINI_BATCH_SIZE * RESPONSE_LEN // self.cp_size
        expected_global_tokens = float(self.dp_size * self.cp_size * tokens_per_cp_rank)

        torch.manual_seed(SEED)
        loss_mat = torch.randn(MINI_BATCH_SIZE, RESPONSE_LEN // self.cp_size, device=self.device)
        loss_mask = torch.ones_like(loss_mat, dtype=torch.bool)
        local_masked_sum = loss_mat.sum().item()

        loss = agg_loss(
            loss_mat=loss_mat,
            loss_mask=loss_mask,
            loss_agg_mode="token-mean",
            dp_size=self.dp_size,
            batch_num_tokens=int(expected_global_tokens),
        )

        expected_loss = local_masked_sum / expected_global_tokens * self.dp_size

        if self.rank == 0:
            print(f"  local_sum           = {local_masked_sum:.6f}")
            print(f"  global_tokens       = {expected_global_tokens}")
            print(f"  dp_size             = {self.dp_size}")
            print(f"  agg_loss result     = {loss.item():.6f}")
            print(f"  expected            = {expected_loss:.6f}")

        torch.testing.assert_close(
            loss,
            torch.tensor(expected_loss, device=self.device),
            atol=1e-5,
            rtol=1e-5,
        )

        if self.rank == 0:
            print("[PASS] test_agg_loss_with_correct_bnt\n")

    # ------------------------------------------------------------------
    # Test 3: engine forward_backward_batch uses DP×CP group
    # ------------------------------------------------------------------
    def test_engine_uses_dp_cp_group(self):
        """MegatronEngine.forward_backward_batch must all-reduce batch_num_tokens
        over the DP×CP group (with_context_parallel=True)."""

        if self.rank == 0:
            print("[TEST 3] engine forward_backward_batch must use DP×CP group")

        import inspect

        from verl.workers.engine.megatron.transformer_impl import MegatronEngine

        src = inspect.getsource(MegatronEngine.forward_backward_batch)

        uses_cp_group = "with_context_parallel=True" in src

        if self.rank == 0:
            if uses_cp_group:
                print("  forward_backward_batch uses with_context_parallel — correct")
            else:
                print("  forward_backward_batch does NOT use with_context_parallel")

        assert uses_cp_group, (
            "MegatronEngine.forward_backward_batch all-reduces batch_num_tokens "
            "over mpu.get_data_parallel_group() (pure DP, without CP). "
            "When CP > 1 this gives a token count that is CP× too small. "
            "Fix: use mpu.get_data_parallel_group(with_context_parallel=True)."
        )

        if self.rank == 0:
            print("[PASS] test_engine_uses_dp_cp_group\n")


if __name__ == "__main__":
    assert int(os.environ.get("WORLD_SIZE", 1)) > 1, (
        "This test must run in distributed mode with torchrun. "
        f"Need at least {ACTOR_TP * ACTOR_PP * ACTOR_CP} GPUs "
        f"(TP={ACTOR_TP} × PP={ACTOR_PP} × CP={ACTOR_CP})."
    )

    test = TestBatchNumTokensCPGroup()
    try:
        test.test_dp_cp_allreduce_gives_global_token_count()
        test.test_agg_loss_with_correct_bnt()
        test.test_engine_uses_dp_cp_group()

        if test.rank == 0:
            print("=" * 60)
            print("ALL TESTS PASSED")
            print("=" * 60)
    finally:
        test.shutdown()
