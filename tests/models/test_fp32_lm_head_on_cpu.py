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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verl.models.transformers.lm_head import (
    fp32_linear,
    install_fp32_lm_head,
)
from verl.models.transformers.monkey_patch import patch_forward_with_backends
from verl.workers.config import RolloutConfig
from verl.workers.engine import EngineRegistry
from verl.workers.rollout.replica import RolloutReplica


@pytest.mark.parametrize("hidden_shape", [(7, 5), (2, 7, 5)])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_fp32_linear_matches_quantized_input_reference(hidden_shape, temperature):
    torch.manual_seed(42)
    vocab_size = 11
    if len(hidden_shape) == 2:
        hidden = torch.randn(hidden_shape[-1], hidden_shape[0]).t()
    else:
        hidden = torch.randn(hidden_shape[0], hidden_shape[-1], hidden_shape[1]).transpose(1, 2)
    hidden = hidden.to(torch.bfloat16).detach().requires_grad_(True)
    weight = torch.randn(vocab_size, hidden_shape[-1]).to(torch.bfloat16).requires_grad_(True)
    bias = torch.randn(vocab_size).to(torch.bfloat16).requires_grad_(True)
    assert not hidden.is_contiguous()

    output = fp32_linear(hidden, weight, bias) / temperature
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    expected_hidden = hidden.detach().float().requires_grad_(True)
    expected_weight = weight.detach().float().requires_grad_(True)
    expected_bias = bias.detach().float().requires_grad_(True)
    expected = torch.nn.functional.linear(expected_hidden, expected_weight, expected_bias) / temperature
    expected.backward(grad_output)

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad.to(torch.bfloat16))
    torch.testing.assert_close(weight.grad, expected_weight.grad.to(torch.bfloat16))
    torch.testing.assert_close(bias.grad, expected_bias.grad.to(torch.bfloat16))

    rounded_after_projection = torch.nn.functional.linear(hidden.detach(), weight.detach(), bias.detach()).float()
    assert not torch.equal(output * temperature, rounded_after_projection)


def test_fp32_linear_only_computes_requested_gradients():
    hidden = torch.randn(4, 3, dtype=torch.bfloat16)
    weight = torch.randn(5, 3, dtype=torch.bfloat16, requires_grad=True)

    fp32_linear(hidden, weight).square().mean().backward()

    assert hidden.grad is None
    assert weight.grad is not None
    assert weight.grad.dtype == weight.dtype


def test_install_fp32_lm_head_preserves_tied_parameter_and_state_dict():
    class TinyTiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(13, 5, dtype=torch.bfloat16)
            self.lm_head = nn.Linear(5, 13, bias=False, dtype=torch.bfloat16)
            self.lm_head.weight = self.embed_tokens.weight

        def get_output_embeddings(self):
            return self.lm_head

    model = TinyTiedModel()
    weight = model.lm_head.weight
    state_dict_keys = tuple(model.state_dict())

    installed = install_fp32_lm_head(model)
    output = installed(torch.randn(2, 3, 5, dtype=torch.bfloat16))

    assert installed is model.lm_head
    assert model.lm_head.weight is weight
    assert model.lm_head.weight is model.embed_tokens.weight
    assert tuple(model.state_dict()) == state_dict_keys
    assert output.dtype == torch.float32


def test_install_fp32_lm_head_rejects_unsupported_output_layer():
    class UnsupportedModel(nn.Module):
        def get_output_embeddings(self):
            return nn.Identity()

    with pytest.raises(TypeError, match="torch.nn.Linear"):
        install_fp32_lm_head(UnsupportedModel())


def test_fp32_lm_head_rejects_architecture_specific_fused_forward():
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_vl")

    with pytest.raises(NotImplementedError, match="qwen3_vl"):
        patch_forward_with_backends(
            model,
            use_fused_kernels=True,
            fused_kernels_backend="torch",
            lm_head_dtype="float32",
        )


def test_non_fsdp_training_backend_rejects_fp32_lm_head():
    with pytest.raises(ValueError, match="FSDP/FSDP2"):
        EngineRegistry.new(
            model_type="language_model",
            backend="megatron",
            model_config=SimpleNamespace(lm_head_dtype="float32"),
        )


def test_non_vllm_rollout_backend_rejects_fp32_lm_head():
    class ConcreteRolloutReplica(RolloutReplica):
        async def launch_servers(self):
            raise NotImplementedError

    replica = object.__new__(ConcreteRolloutReplica)
    with pytest.raises(ValueError, match="vLLM rollout"):
        RolloutReplica.__init__(
            replica,
            replica_rank=0,
            config=RolloutConfig(name="sglang"),
            model_config=SimpleNamespace(lm_head_dtype="float32"),
        )
