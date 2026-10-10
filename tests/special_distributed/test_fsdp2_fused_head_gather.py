# Copyright 2026 verl contributors
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

"""Run with torchrun --standalone --nproc-per-node=2 -m pytest -x -q <this file>.

Use four ranks to exercise a two-dimensional HSDP mesh as well.
"""

import copy
import os
from types import MethodType

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.models.transformers.dense_common import forward_with_torch_backend, forward_with_triton_backend
from verl.utils.experimental import torch_functional as experimental_F
from verl.utils.fsdp_utils import apply_fsdp2, set_fsdp2_gradient_sync
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy


def _assert_close_on_all_ranks(actual, expected, **kwargs):
    error = None
    try:
        torch.testing.assert_close(actual, expected, **kwargs)
    except AssertionError as exc:
        error = exc
    failed = torch.tensor(int(error is not None), device="cuda")
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        if error is not None:
            raise error
        pytest.fail("Numerical comparison failed on another rank")


def _reference_head_forward(self, hidden, labels, temperature):
    assert not isinstance(self.weight, DTensor)
    if self._backend == "triton":
        return linear_cross_entropy(hidden, self.weight, labels, temperature)
    return experimental_F.FusedLinearForPPO(impl_backend=self._backend)(hidden, self.weight, labels, temperature)


def _reference_model_forward(self, input_ids, temperature):
    hidden = self.model(input_ids, use_cache=False, return_dict=True)[0]
    return self.lm_head(hidden, torch.roll(input_ids, -1, -1), temperature)


@pytest.fixture(scope="module")
def mesh():
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("requires torchrun with at least two CUDA ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if world_size == 4:
        yield init_device_mesh("cuda", (2, 2), mesh_dim_names=("replicate", "shard"))
    else:
        yield init_device_mesh("cuda", (world_size,))
    dist.destroy_process_group()


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_weight_materialization_preserves_autograd(mesh, offload, sharded):
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(42)
    original = torch.randn(32, 16, device=device)
    if sharded:
        weight = distribute_tensor(original, mesh, [Replicate()] * (mesh.ndim - 1) + [Shard(0)])
        if offload:
            weight = weight.cpu()
        weight = weight.detach().requires_grad_()
    else:
        weight = original.to("cpu" if offload else device).detach().requires_grad_()
    torch.manual_seed(100 + dist.get_rank())
    hidden = torch.randn(8, 16, device=device, requires_grad=True)
    materialized = experimental_F.prepare_fused_linear_weight(hidden, weight)
    assert not isinstance(materialized, DTensor)
    assert materialized.device == hidden.device
    assert materialized.requires_grad
    torch.testing.assert_close(materialized, original)
    if not sharded and not offload:
        assert materialized is weight

    (hidden @ materialized.t()).square().mean().backward()
    reference = original.detach().requires_grad_()
    (hidden.detach() @ reference.t()).square().mean().backward()
    if sharded:
        dist.all_reduce(reference.grad, op=dist.ReduceOp.AVG)
        actual_grad = weight.grad.to(device).full_tensor()
    else:
        actual_grad = weight.grad.to(device)
    torch.testing.assert_close(actual_grad, reference.grad)
    assert weight.device.type == ("cpu" if offload else "cuda")
    assert weight.grad.device == weight.device


@pytest.mark.parametrize("backend", ["torch", "triton", "liger"])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("no_sync", [False, True])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen3_fsdp2_fused_head_training(mesh, backend, offload, no_sync, tied, dtype):
    if backend == "liger":
        assert experimental_F._LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY is not None, "install liger-kernel"
        if dtype == torch.float32 and torch.cuda.get_device_capability()[0] == 9:
            pytest.skip("Liger's SM90 fused scaled cross entropy supports BF16 only")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
        use_cache=False,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config).to(device)
    reference = copy.deepcopy(model)
    # Use the same kernel inside the reference head's normal FSDP lifecycle,
    # isolating materialization/gradient correctness from kernel rounding.
    reference.lm_head._backend = backend
    reference.lm_head.forward = MethodType(_reference_head_forward, reference.lm_head)
    reference.forward = MethodType(_reference_model_forward, reference)
    model._verl_fused_kernels_backend = backend
    forward = forward_with_triton_backend if backend == "triton" else forward_with_torch_backend
    model.forward = MethodType(forward, model)
    kwargs = {"mesh": mesh, "mp_policy": MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)}
    if offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()
    apply_fsdp2(model, kwargs, {})
    apply_fsdp2(reference, kwargs, {})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
    assert isinstance(model.lm_head.weight, DTensor)
    assert model.lm_head.weight.dtype == torch.float32
    assert model.lm_head.weight.device.type == ("cpu" if offload else "cuda")
    tolerance = {"atol": 2e-3, "rtol": 2e-2} if dtype == torch.bfloat16 else {"atol": 2e-5, "rtol": 2e-4}

    torch.manual_seed(100 + dist.get_rank())
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            sync = not no_sync or micro == 1
            set_fsdp2_gradient_sync(model, sync)
            reference.set_requires_gradient_sync(sync)
            inputs = torch.randint(config.vocab_size, (2, 8), device=device)
            labels = torch.roll(inputs, -1, -1)
            scales = torch.rand(2, 8, device=device)
            temperature = 0.7
            expected_log_probs, expected_entropy = reference(inputs, temperature=temperature)
            expected_log_probs = expected_log_probs.reshape_as(labels)
            expected_entropy = expected_entropy.reshape_as(labels)
            actual = model(inputs, return_dict=True, temperature=temperature)
            actual_log_probs = actual.log_probs.reshape_as(labels)
            actual_entropy = actual.entropy.reshape_as(labels)
            _assert_close_on_all_ranks(actual_log_probs, expected_log_probs, check_dtype=False, **tolerance)
            _assert_close_on_all_ranks(actual_entropy, expected_entropy, check_dtype=False, **tolerance)
            expected_loss = -((expected_log_probs + 0.05 * expected_entropy) * scales).mean() / 2
            actual_loss = -((actual_log_probs + 0.05 * actual_entropy) * scales).mean() / 2
            expected_loss.backward()
            actual_loss.backward()
            set_fsdp2_gradient_sync(model, True)
            reference.set_requires_gradient_sync(True)

        for (name, param), (ref_name, ref_param) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == ref_name
            assert param.grad is not None, f"missing gradient for {name}, step {step}"
            assert param.grad.device == param.device
            _assert_close_on_all_ranks(
                param.grad.to_local().to(device),
                ref_param.grad.to_local().to(device),
                msg=lambda message, name=name, step=step: f"{name}, step {step}: {message}",
                **tolerance,
            )
        optimizer.step()
        reference_optimizer.step()
        for param, ref_param in zip(model.parameters(), reference.parameters(), strict=True):
            _assert_close_on_all_ranks(param.to_local().to(device), ref_param.to_local().to(device), **tolerance)
