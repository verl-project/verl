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
"""End-to-end regression test for issue #6068 (use_fused_kernels=True + ulysses_sp>1).

Construction:
1. Build a small Llama-style model patched with `use_fused_kernels=True` and SP=2.
2. Run the fused forward through Ulysses (rank-local input_ids_rmpad slice + the
   engine-style globally-rolled `shift_labels`) and compare its `log_probs`
   against a single-GPU reference (SP=1, same fused kernel) gathered across ranks.
3. Without the fix, every rank's last-position log_prob diverges from the
   reference because the fused forward locally re-rolls the SP-sliced input.
   With the fix, log_probs are bitwise close.

Run on 2 GPUs:
    torchrun --nproc_per_node=2 -m pytest -svv \
        tests/special_distributed/test_fused_kernels_ulysses_sp.py
"""

import pytest
import torch
import torch.distributed
from torch.distributed import init_device_mesh
from transformers import AutoModelForCausalLM, LlamaConfig

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.utils.ulysses import (
    FSDPUlyssesShardingManager,
    gather_outputs_and_unpad,
    set_ulysses_sequence_parallel_group,
    ulysses_pad_and_slice_inputs,
)


def _make_model(seed: int = 0):
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
    )
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    return model, config


def _broadcast_params(model):
    for p in model.parameters():
        torch.distributed.broadcast(p.data, src=0)


@pytest.fixture(scope="module", autouse=True)
def _init_dist():
    if not torch.distributed.is_initialized():
        initialize_global_process_group()
    yield


def test_fused_kernels_log_probs_match_under_sp():
    """log_probs from fused-kernel + SP=2 must equal fused-kernel + SP=1.

    This is the direct regression test for #6068. The bug was that the
    fused-forward functions re-rolled the SP-sliced `input_ids` locally,
    producing wrong labels at the shard boundary. The fix passes
    `shift_labels` (engine-rolled, then SP-sliced) so the fused forward
    uses correct labels.
    """
    assert get_torch_device().device_count() >= 2, "need at least 2 gpus"
    sp_size = 2
    device = get_device_name()

    # Build identical model on every rank.
    model, config = _make_model(seed=42)
    apply_monkey_patch(
        model,
        ulysses_sp_size=sp_size,
        use_remove_padding=True,
        use_fused_kernels=True,
        fused_kernels_backend="torch",
    )
    model = model.to(device=device, dtype=torch.bfloat16)
    _broadcast_params(model)

    # rank-0-authoritative input. total_nnz must be divisible by sp_size.
    total_nnz = 32
    if torch.distributed.get_rank() == 0:
        input_ids = torch.randint(0, config.vocab_size, (1, total_nnz), device=device)
    else:
        input_ids = torch.empty((1, total_nnz), dtype=torch.long, device=device)
    torch.distributed.broadcast(input_ids, src=0)
    position_ids = torch.arange(total_nnz, device=device).unsqueeze(0)  # (1, total_nnz)

    # ---- SP=1 reference: full sequence, fused forward, on rank 0 only ----
    set_ulysses_sequence_parallel_group(None)
    if torch.distributed.get_rank() == 0:
        ref_out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            temperature=1.0,
        )
        ref_log_probs = ref_out.log_probs.detach().clone()  # (1, total_nnz)
    else:
        ref_log_probs = torch.empty((1, total_nnz), device=device, dtype=torch.float32)
    torch.distributed.broadcast(ref_log_probs, src=0)

    # ---- SP=2 path: each rank gets a slice; engine plumbs shift_labels ----
    world_size = torch.distributed.get_world_size()
    assert world_size % sp_size == 0
    dp_size = world_size // sp_size
    device_mesh = init_device_mesh(device_type=device, mesh_shape=(dp_size, sp_size), mesh_dim_names=("dp", "sp"))
    sharding_manager = FSDPUlyssesShardingManager(device_mesh)

    with sharding_manager:
        # Mimic the engine's preparation (transformer_impl.py ~951-984).
        input_ids_rmpad = input_ids  # (1, total_nnz), already "rmpadded"
        input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
        # Slice both inputs and rolled labels exactly like the engine.
        # `position_ids` is already (1, total_nnz); helper expects 2D, not 3D.
        sliced_input, sliced_pos, pad_size = ulysses_pad_and_slice_inputs(
            input_ids_rmpad,
            position_ids_rmpad=position_ids,
            sp_size=sp_size,
        )
        sliced_rolled, _, _ = ulysses_pad_and_slice_inputs(
            input_ids_rmpad_rolled, position_ids_rmpad=None, sp_size=sp_size
        )
        sp_out = model(
            input_ids=sliced_input,
            position_ids=sliced_pos,
            use_cache=False,
            return_dict=True,
            temperature=1.0,
            shift_labels=sliced_rolled,
        )
        # log_probs is (1, total_nnz/sp + pad). Gather across SP ranks → (1, total_nnz).
        sp_log_probs_local = sp_out.log_probs  # (1, local_len)
        sp_log_probs = gather_outputs_and_unpad(
            sp_log_probs_local.squeeze(0),
            gather_dim=0,
            unpad_dim=0,
            padding_size=pad_size,
        ).unsqueeze(0)  # (1, total_nnz)

    # bf16 matmul + chunked fused kernel — keep tolerances loose enough to absorb
    # numerical noise but strict enough to catch the off-by-one label bug, which
    # produces O(1) errors in the affected positions.
    torch.testing.assert_close(sp_log_probs, ref_log_probs, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-svv"])
