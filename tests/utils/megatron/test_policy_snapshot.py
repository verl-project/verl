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

"""Decoupled PPO must restore the policy's persistent buffers as well as parameters."""

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

from verl.utils.megatron_utils import (
    copy_megatron_model_to_cpu,
    load_megatron_model_to_gpu,
    offload_megatron_model_to_cpu,
    restore_megatron_model_from_cpu,
)


@pytest.fixture
def model_parallel(tmp_path):
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


class RoutedPolicy(torch.nn.Module):
    """A real Megatron router with distinct expert outputs, without a pretrained checkpoint."""

    def __init__(self, config, pg):
        super().__init__()
        self.config = config
        self.router = TopKRouter(config, pg_collection=pg)
        self.expert_logits = torch.nn.Parameter(3 * torch.eye(4))
        # The margin is smaller than one default bias-update step, with no argmax ties.
        with torch.no_grad():
            self.router.weight.zero_()
            self.router.weight[:, 0] = torch.tensor([0.0012, 0.0008, 0.0004, 0.0])

    def forward(self, inputs):
        probabilities, routes = self.router(inputs)
        return (probabilities @ self.expert_logits).log_softmax(-1), routes


@pytest.mark.parametrize(
    ("wrapped", "offloaded", "bias_update_rate"),
    [
        (False, False, 0.001),
        (True, False, 0.001),
        (True, True, 0.001),
        (False, False, 0.0),
        (True, False, 0.0),
        (True, True, 0.0),
    ],
    ids=["unwrapped", "ddp", "bf16-offloaded", "unwrapped-frozen", "ddp-frozen", "bf16-offloaded-frozen"],
)
def test_snapshot_restores_old_and_current_moe_policy(model_parallel, wrapped, offloaded, bias_update_rate):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_moe_experts=4,
        moe_router_topk=1,
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type="none",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=bias_update_rate,
        moe_router_fusion=False,
        add_bias_linear=False,
        bf16=offloaded,
        params_dtype=torch.bfloat16 if offloaded else torch.float32,
    )
    raw = RoutedPolicy(config, model_parallel).cuda()
    model = Float16Module(config, raw) if offloaded else raw
    if wrapped:
        model = DDP(config, DistributedDataParallelConfig(use_distributed_optimizer=True), model)
    inputs = torch.zeros(8, 1, 16, device="cuda")
    inputs[..., 0] = 1
    with torch.no_grad():
        old_log_probs, old_routes = model(inputs)
    old_bias = raw.router.expert_bias.clone()
    old_parameters = [p.detach().clone() for p in raw.parameters()]
    bias_pointer = raw.router.expert_bias.data_ptr()

    def offload_and_check():
        offload_megatron_model_to_cpu([model])
        buffers = model.buffers + model.expert_parallel_buffers
        assert buffers
        for buffer in buffers:
            assert buffer.param_data.untyped_storage().nbytes() == 0
            assert buffer.param_data.cpu_data.device.type == "cpu"
            assert buffer.param_data.cpu_data.dtype == torch.bfloat16
        # Parameter offload must leave model buffers on the compute device.
        assert raw.router.expert_bias.device == inputs.device
        assert raw.router.expert_bias.dtype == torch.float32

    if offloaded:
        offload_and_check()
    old_state = copy_megatron_model_to_cpu([model])

    # Collect routing counts before updating the bias.
    if offloaded:
        load_megatron_model_to_gpu([model])
    model(inputs)
    _update_router_expert_bias([model], config, tp_dp_cp_group=model_parallel.tp_dp_cp)
    with torch.no_grad():
        raw.expert_logits.mul_(1.1)
        current_log_probs, current_routes = model(inputs)
    current_bias = raw.router.expert_bias.clone()
    current_parameters = [p.detach().clone() for p in raw.parameters()]
    if offloaded:
        offload_and_check()
    current_state = copy_megatron_model_to_cpu([model])
    if bias_update_rate > 0:
        assert not torch.equal(old_routes, current_routes), "The stimulus must change the routing decision"
        assert not torch.equal(old_bias, current_bias)
    else:
        # Freezing removes this trigger, while parameter updates still require a snapshot.
        assert torch.equal(old_routes, current_routes)
        assert torch.equal(old_bias, current_bias)
    assert not torch.equal(old_log_probs, current_log_probs)

    # Runtime counters do not belong to either policy snapshot.
    raw.router.local_tokens_per_expert.fill_(123)
    for _ in range(2):
        for state, bias, parameters, log_probs, routes in (
            (old_state, old_bias, old_parameters, old_log_probs, old_routes),
            (current_state, current_bias, current_parameters, current_log_probs, current_routes),
        ):
            restore_megatron_model_from_cpu([model], state)
            assert torch.equal(raw.router.expert_bias, bias), "policy snapshot did not restore expert_bias"
            assert raw.router.expert_bias.data_ptr() == bias_pointer
            if offloaded:
                # Match the worker: restore the offloaded snapshot before eval loads params.
                assert all(buffer.param_data.untyped_storage().nbytes() == 0 for buffer in model.buffers)
                load_megatron_model_to_gpu([model], load_grad=False)
            for parameter, expected in zip(raw.parameters(), parameters, strict=True):
                assert torch.equal(parameter, expected)
            with torch.no_grad():
                restored_log_probs, restored_routes = model(inputs)
            assert torch.equal(restored_routes, routes)
            torch.testing.assert_close(restored_log_probs, log_probs, rtol=0, atol=0)
            assert torch.all(raw.router.local_tokens_per_expert == 123)
            if offloaded:
                offload_and_check()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_snapshot_buffer_storage_and_lifetime(model_parallel, device):
    models = [torch.nn.Linear(2, 2, device=device), torch.nn.Linear(2, 2, device=device)]
    expected = []
    pointers = []
    for index, model in enumerate(models):
        model.register_buffer("counter", torch.tensor(index + 3, dtype=torch.int64, device=device))
        model.register_buffer("scratch", torch.zeros(2, device=device), persistent=False)
        model.register_buffer("optional", None)
        model.child = torch.nn.Module()
        model.child.register_buffer("scale", torch.tensor([1.5 + index], dtype=torch.bfloat16, device=device))
        model.register_buffer("scale_alias", model.child.scale)
        expected.append((model.counter.clone(), model.child.scale.clone()))
        pointers.append((model.counter.data_ptr(), model.child.scale.data_ptr()))

    snapshot = copy_megatron_model_to_cpu(models)
    for model in models:
        model.counter.add_(10)
        model.child.scale.mul_(2)
        # Forward caches may change shape; restoring a policy must not rewind them.
        model.scratch = torch.ones(7, device=device)

    restore_megatron_model_from_cpu(models, snapshot)
    for model, (counter, scale), (counter_ptr, scale_ptr) in zip(models, expected, pointers, strict=True):
        assert torch.equal(model.counter, counter), "policy snapshot did not restore persistent counter"
        assert torch.equal(model.child.scale, scale)
        assert model.counter.dtype == torch.int64
        assert model.child.scale.dtype == torch.bfloat16
        assert model.counter.data_ptr() == counter_ptr
        assert model.child.scale.data_ptr() == scale_ptr
        assert model.scale_alias is model.child.scale
        assert torch.equal(model.scratch, torch.ones(7, device=device))
        assert model.optional is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("transient_first", [False, True], ids=["persistent-first", "transient-first"])
def test_snapshot_preserves_mixed_persistence_aliases(model_parallel, device, transient_first):
    model = torch.nn.Module()
    shared = torch.tensor([1.25], device=device)
    registrations = [("transient_alias", False), ("persistent_state", True)]
    if not transient_first:
        registrations.reverse()
    for name, persistent in registrations:
        model.register_buffer(name, shared, persistent=persistent)
    model.register_buffer("independent_transient", torch.zeros(1, device=device), persistent=False)
    expected = model.persistent_state.clone()
    pointer = shared.data_ptr()
    assert set(model.state_dict()) == {"persistent_state"}

    snapshot = copy_megatron_model_to_cpu([model])
    shared.add_(10)
    model.independent_transient.add_(7)
    restore_megatron_model_from_cpu([model], snapshot)

    assert torch.equal(model.persistent_state, expected), "policy snapshot omitted persistent alias"
    assert model.persistent_state is model.transient_alias is shared
    assert shared.data_ptr() == pointer
    assert torch.equal(model.independent_transient, torch.full_like(shared, 7))
    assert set(snapshot["model_chunk_0"]["persistent_buffers"]) == {"persistent_state"}
