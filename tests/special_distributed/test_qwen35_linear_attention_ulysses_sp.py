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
"""Qwen3.5 linear-attention regression test for packed Ulysses SP.

Run on GPUs, for example:
    torchrun --nproc_per_node=8 -m pytest -svv \
        tests/special_distributed/test_qwen35_linear_attention_ulysses_sp.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from verl.models.transformers.qwen3_5 import qwen3_5_decoder_layer_forward, qwen3_5_gated_delta_net_forward
from verl.utils.ulysses import set_ulysses_sequence_parallel_group

pytest.importorskip("fla")
qwen35_config_mod = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
qwen35_modeling_mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")


Qwen3_5TextConfig = qwen35_config_mod.Qwen3_5TextConfig
Qwen3_5DecoderLayer = qwen35_modeling_mod.Qwen3_5DecoderLayer
Qwen3_5GatedDeltaNet = qwen35_modeling_mod.Qwen3_5GatedDeltaNet

pytestmark = pytest.mark.skipif("LOCAL_RANK" not in os.environ, reason="run with torchrun")

PROBE_HIDDEN_SIZE = 512
PROBE_HEAD_DIM = 128


def _err_ratio(ref: torch.Tensor, actual: torch.Tensor) -> float:
    err = (ref.detach() - actual.detach()).flatten().float().square().mean().sqrt().item()
    base = ref.detach().flatten().float().square().mean().sqrt().item()
    return err / (base + 1e-8)


def _make_layer(device: torch.device) -> Qwen3_5DecoderLayer:
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=PROBE_HIDDEN_SIZE,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=PROBE_HEAD_DIM,
        linear_key_head_dim=PROBE_HEAD_DIM,
        linear_value_head_dim=PROBE_HEAD_DIM,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
        dtype=torch.bfloat16,
    )
    torch.manual_seed(1234)
    layer = Qwen3_5DecoderLayer(config, 0).to(device=device, dtype=torch.bfloat16)
    layer.eval()
    return layer


def _broadcast_params(model: torch.nn.Module):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def _all_gather_seq(x: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=1)


def _cases_for_world_size(world_size: int) -> list[tuple[str, list[int]]]:
    total = 8 * world_size
    if world_size == 2:
        return [
            ("boundary_aligned", [8, 8]),
            ("sequence_cut", [6, 10]),
            ("single_sequence", [16]),
            ("many_short_sequences", [3, 4, 5, 4]),
        ]
    if world_size == 4:
        return [
            ("boundary_aligned", [8, 8, 8, 8]),
            ("sequence_cut", [20, 12]),
            ("single_sequence", [32]),
            ("many_short_sequences", [5, 7, 9, 11]),
        ]
    if world_size == 8:
        return [
            ("boundary_aligned", [128, 128, 128, 128, 128, 128, 128, 128]),
            ("sequence_cut", [700, 324]),
            ("single_sequence", [1024]),
            ("many_short_sequences", [100, 150, 200, 250, 124, 100, 100]),
        ]
    return [("single_sequence", [total])]


def _run_case(case_name: str, lengths: list[int], layer: Qwen3_5DecoderLayer, device: torch.device):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    total_tokens = sum(lengths)
    if total_tokens % world_size != 0:
        raise RuntimeError(f"{case_name}: total tokens {total_tokens} must divide world_size {world_size}.")

    cu_seqlens = torch.tensor([0] + torch.cumsum(torch.tensor(lengths), 0).tolist(), device=device, dtype=torch.long)
    cu_seqlens_cpu = cu_seqlens.cpu()
    position_embeddings = (torch.empty(0, device=device), torch.empty(0, device=device))

    torch.manual_seed(5678 + total_tokens + len(lengths))
    full_hidden = torch.randn(1, total_tokens, PROBE_HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    set_ulysses_sequence_parallel_group(None)
    full_hidden_ref = full_hidden.detach().clone().requires_grad_(True)
    ref_out = layer(
        full_hidden_ref,
        position_embeddings=position_embeddings,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    ref_out.sum().backward()
    ref_grad = full_hidden_ref.grad.detach()
    layer.zero_grad(set_to_none=True)

    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    local_seq_len = total_tokens // world_size
    start = rank * local_seq_len
    local_hidden = full_hidden[:, start : start + local_seq_len].detach().clone().requires_grad_(True)
    sp_out_local = layer(
        local_hidden,
        position_embeddings=position_embeddings,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    sp_out_local.sum().backward()

    sp_out = _all_gather_seq(sp_out_local.detach())
    sp_grad = _all_gather_seq(local_hidden.grad.detach())

    max_out_diff = (sp_out - ref_out.detach()).abs().max().item()
    max_grad_diff = (sp_grad - ref_grad).abs().max().item()
    out_err_ratio = _err_ratio(ref_out.detach(), sp_out)
    grad_err_ratio = _err_ratio(ref_grad, sp_grad)
    if rank == 0:
        print(
            f"{case_name}: max_out_diff={max_out_diff:.6f} max_grad_diff={max_grad_diff:.6f} "
            f"out_err_ratio={out_err_ratio:.6f} grad_err_ratio={grad_err_ratio:.6f}"
        )

    torch.testing.assert_close(sp_out, ref_out.detach(), atol=2e-2, rtol=2e-2)
    assert grad_err_ratio < 2e-3


@pytest.mark.parametrize(
    "use_causal_conv1d_fn",
    [True, False],
    ids=["causal_conv1d_fn", "torch_conv1d_fallback"],
)
def test_qwen35_linear_attention_matches_full_forward_under_ulysses_sp(use_causal_conv1d_fn: bool):
    assert torch.cuda.is_available(), "CUDA is required"
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    Qwen3_5DecoderLayer.forward = qwen3_5_decoder_layer_forward
    Qwen3_5GatedDeltaNet.forward = qwen3_5_gated_delta_net_forward

    layer = _make_layer(device)
    if not use_causal_conv1d_fn:
        for module in layer.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.causal_conv1d_fn = None
    _broadcast_params(layer)

    for case_name, lengths in _cases_for_world_size(dist.get_world_size()):
        _run_case(case_name, lengths, layer, device)

    set_ulysses_sequence_parallel_group(None)
