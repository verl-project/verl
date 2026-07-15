# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Four-rank NCCL regression for Dynamic CP forward and reverse routing."""

import os

import torch
import torch.distributed as dist

from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.dynamic_cp_scheduler import _reroute_samples, reverse_route_outputs


def _nested(parts: list[torch.Tensor]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(parts, layout=torch.jagged)


def main() -> None:
    if not torch.cuda.is_available():
        # Dynamic CP is only qualified on CUDA/NCCL. run_all.sh is shared with
        # the Ascend NPU workflow, so skip instead of failing there.
        print("Skipping the Dynamic CP routing regression: CUDA is unavailable")
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4, f"This regression requires four ranks, got {world_size}"

    # CP-fast layout for DP2 x CP2: each static DP group is one CP column.
    dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    dp_group = dp_groups[rank % 2]
    dcp_group = dist.group.WORLD

    # DP rank 0 owns one sample, while DP rank 1 is intentionally empty. Both
    # static CP replicas participate, matching the engine's replicated inputs.
    owns_sample = rank < 2
    local_samples = [{"input_ids": torch.arange(9, device="cuda", dtype=torch.int64)}] if owns_sample else []
    global_ids_this_rank = (
        torch.tensor([0], device="cuda", dtype=torch.int32)
        if owns_sample
        else torch.empty(0, device="cuda", dtype=torch.int32)
    )
    sample_id_groups = [[[0], [0], [0], [0]]]
    offsets = torch.tensor([0, 1, 1], dtype=torch.int32)

    routed = _reroute_samples(
        local_samples=local_samples,
        global_ids_this_rank=global_ids_this_rank,
        sample_id_groups=sample_id_groups,
        offsets=offsets,
        dp_group=dp_group,
        dcp_group=dcp_group,
        tensor_keys=["input_ids"],
        scalar_keys=[],
        key_dtypes={"input_ids": torch.int64},
    )
    assert routed[0]["input_ids"].dtype == torch.int64
    torch.testing.assert_close(routed[0]["input_ids"], torch.arange(9, device="cuda"))

    # Each dynamic CP rank returns its compact zig-zag token shard. Router
    # replay metadata is already a replicated full-sequence output and must not
    # be reconstructed with the compact indices or summed across replicas.
    compact_indices = [
        torch.tensor([0, 1], device="cuda"),
        torch.tensor([2, 3], device="cuda"),
        torch.tensor([4, 5], device="cuda"),
        torch.tensor([6, 7, 8], device="cuda"),
    ][rank]
    model_output = {
        "log_probs": _nested([compact_indices.to(torch.float32)]),
        "routed_experts": _nested([torch.arange(9, device="cuda", dtype=torch.int64).reshape(9, 1, 1)]),
        "_dcp_local_token_indices": _nested([compact_indices]),
        "_dcp_full_seq_lens": _nested([torch.tensor([9], device="cuda", dtype=torch.int64)]),
    }
    routing_info = {
        "sample_id_groups": sample_id_groups,
        "offsets": offsets,
        "global_ids_this_rank": global_ids_this_rank,
    }
    restored = reverse_route_outputs(
        model_output,
        routing_info,
        dp_group=dp_group,
        dcp_group=dcp_group,
        merge_duplicate_gids=True,
    )

    if rank == 0:
        torch.testing.assert_close(restored["log_probs"].unbind()[0], torch.arange(9, device="cuda").float())
        torch.testing.assert_close(
            restored["routed_experts"].unbind()[0],
            torch.arange(9, device="cuda", dtype=torch.int64).reshape(9, 1, 1),
        )
    elif rank == 2:
        # Rank 2 is the canonical CP rank for the intentionally empty DP owner.
        # It must not inherit the scheduled sample as a fallback result.
        assert len(restored["log_probs"].unbind()) == 0
        assert len(restored["routed_experts"].unbind()) == 0

    # In Megatron's per-token regime DDP performs a SUM, then
    # finalize_model_grads divides by the all-reduced token count. The uneven
    # 9-token CP4 split (2/2/2/3) exercises DCP's exact local ownership count.
    parameter = torch.tensor(2.0, device="cuda", requires_grad=True)
    coefficients = torch.arange(1, 10, device="cuda", dtype=torch.float32).unsqueeze(0)
    local_mask = torch.zeros_like(coefficients, dtype=torch.bool)
    local_mask[0, compact_indices] = True
    local_loss = agg_loss(
        parameter * coefficients,
        local_mask,
        "token-mean",
        dp_size=dp_group.size(),
        batch_num_tokens=coefficients.numel(),
    )
    global_routed_num_tokens = coefficients.numel()
    local_sum = local_loss * global_routed_num_tokens / dp_group.size()
    local_sum.backward()
    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    local_num_tokens = torch.tensor(compact_indices.numel(), dtype=torch.int, device="cuda")
    dist.all_reduce(local_num_tokens, op=dist.ReduceOp.SUM)
    parameter.grad /= local_num_tokens
    torch.testing.assert_close(parameter.grad, coefficients.mean())

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
