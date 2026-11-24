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
"""Two identical Qwen3.5 gated-delta-net forwards must be bitwise equal on Blackwell.

flash-linear-attention < 0.6.0 lets the autotuner pick 8-warp ``chunk_fwd_kernel_o`` and
``chunk_bwd_kernel_dv_local`` configs that race on Blackwell with Triton < 3.8
(fla-org/flash-linear-attention#1228 and #1327, fixed in 0.6.0). The log-prob pass and
the training pass then see different linear-attention outputs for the same tokens and
weights, so on-policy ``ppo_kl`` is not zero.
"""

import pytest
import torch
import triton
from packaging.version import Version

from verl.models.transformers.qwen3_5 import qwen3_5_gated_delta_net_forward

pytest.importorskip("fla")
qwen35_config_mod = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
qwen35_modeling_mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="the chunk_fwd_kernel_o race needs a Blackwell (SM100+) GPU",
)

NUM_FORWARDS = 10


def _make_gdn(device: torch.device):
    # Qwen3.5-4B linear-attention geometry (16 key heads, 32 value heads, head dim 128).
    config = qwen35_config_mod.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=1024,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
        dtype=torch.bfloat16,
    )
    torch.manual_seed(1234)
    gdn = qwen35_modeling_mod.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(device=device, dtype=torch.bfloat16)
    return gdn.eval()


def test_qwen3_5_gdn_forward_is_bitwise_repeatable_on_blackwell():
    device = torch.device("cuda")
    gdn = _make_gdn(device)

    # One packed micro-batch, as verl feeds it with use_remove_padding=True.
    lengths = [3000, 1500, 4096, 700, 2500, 4096, 1200, 3800, 2900, 4000, 2240, 2742]
    cu_seqlens = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device=device, dtype=torch.long)
    torch.manual_seed(5678)
    hidden_states = torch.randn(1, sum(lengths), 1024, device=device, dtype=torch.bfloat16)

    outputs = []
    with torch.no_grad():
        for _ in range(NUM_FORWARDS):
            out = qwen3_5_gated_delta_net_forward(
                gdn, hidden_states, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens.cpu()
            )
            outputs.append(out.clone())

    mismatched = [i for i, out in enumerate(outputs[1:], start=1) if not torch.equal(out, outputs[0])]
    max_diff = max(((out.float() - outputs[0].float()).abs().max().item() for out in outputs[1:]), default=0.0)
    assert not mismatched, (
        f"forwards {mismatched} differ from forward 0 (max abs diff {max_diff:.3e}); "
        "flash-linear-attention < 0.6.0 races on Blackwell, see fla-org/flash-linear-attention#1228"
    )


def _autotuner(kernel):
    while not hasattr(kernel, "configs"):
        kernel = kernel.fn
    return kernel


@pytest.mark.skipif(Version(triton.__version__) >= Version("3.8.0"), reason="Triton >= 3.8 has the fix")
def test_fla_offers_no_racy_8_warp_configs_on_blackwell():
    chunk_o = pytest.importorskip("fla.ops.common.chunk_o")
    for name in ("chunk_fwd_kernel_o", "chunk_bwd_kernel_dv_local"):
        warps = sorted({config.num_warps for config in _autotuner(getattr(chunk_o, name)).configs})
        assert max(warps) <= 4, (
            f"{name} offers num_warps={warps} on Blackwell with Triton {triton.__version__}; "
            "use flash-linear-attention >= 0.6.0 (fla-org/flash-linear-attention#1228, #1327)"
        )
