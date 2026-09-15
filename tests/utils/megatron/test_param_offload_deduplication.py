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
from unittest.mock import create_autospec

import pytest
import torch

from verl.utils import megatron_utils


def _non_owner_group():
    return megatron_utils.MegatronParamOffloadGroup(process_group=object(), source_rank=0, is_owner=False)


def _owner_group():
    return megatron_utils.MegatronParamOffloadGroup(process_group=object(), source_rank=0, is_owner=True)


def test_param_offload_group_builder_uses_dense_and_expert_replica_groups(monkeypatch):
    dense_process_group = object()
    expert_process_group = object()
    get_dense_group = create_autospec(
        lambda with_context_parallel, with_gtp_remat=True: None, return_value=dense_process_group
    )
    get_expert_group = create_autospec(lambda with_gtp_remat=True: None, return_value=expert_process_group)
    monkeypatch.setattr(megatron_utils.mpu, "get_data_parallel_group", get_dense_group)
    monkeypatch.setattr(megatron_utils.mpu, "get_expert_data_parallel_group", get_expert_group)
    monkeypatch.setattr(megatron_utils, "_make_param_offload_group", lambda process_group: process_group)

    groups = megatron_utils.build_megatron_param_offload_groups(enabled=True)

    assert groups.dense is dense_process_group
    assert groups.expert is expert_process_group
    get_dense_group.assert_called_once_with(with_context_parallel=True, with_gtp_remat=False)
    get_expert_group.assert_called_once_with(with_gtp_remat=False)


def test_param_offload_group_builder_supports_mcore_without_gtp_remat(monkeypatch):
    dense_process_group = object()
    expert_process_group = object()
    get_dense_group = create_autospec(lambda with_context_parallel: None, return_value=dense_process_group)
    get_expert_group = create_autospec(lambda: None, return_value=expert_process_group)
    monkeypatch.setattr(megatron_utils.mpu, "get_data_parallel_group", get_dense_group)
    monkeypatch.setattr(megatron_utils.mpu, "get_expert_data_parallel_group", get_expert_group)
    monkeypatch.setattr(megatron_utils, "_make_param_offload_group", lambda process_group: process_group)

    groups = megatron_utils.build_megatron_param_offload_groups(enabled=True)

    assert groups.dense is dense_process_group
    assert groups.expert is expert_process_group
    get_dense_group.assert_called_once_with(with_context_parallel=True)
    get_expert_group.assert_called_once_with()


def test_param_offload_group_builder_returns_none_without_replicas(monkeypatch):
    monkeypatch.setattr(megatron_utils, "_get_megatron_param_replica_process_groups", lambda: (object(), object()))
    monkeypatch.setattr(megatron_utils, "_make_param_offload_group", lambda process_group: None)

    assert megatron_utils.build_megatron_param_offload_groups(enabled=True) is None


def test_non_owner_ddp_buffer_keeps_no_cpu_copy_and_restores_from_broadcast(monkeypatch):
    expected = torch.arange(12, dtype=torch.float32)
    buffer = SimpleNamespace(param_data=expected.clone())
    group = _non_owner_group()

    megatron_utils._offload_ddp_param_buffer(buffer, group)

    assert buffer.param_data.storage().size() == 0
    assert buffer.param_data.cpu_data is None
    monkeypatch.setattr(
        megatron_utils,
        "_broadcast_offloaded_parameter",
        lambda tensor, unused_group: tensor.copy_(expected),
    )
    megatron_utils._load_ddp_param_buffer(buffer, group)
    torch.testing.assert_close(buffer.param_data, expected)


@pytest.mark.parametrize("group", [None, _owner_group()], ids=["default", "owner"])
def test_ddp_owner_reuses_cpu_copy_across_round_trip(monkeypatch, group):
    torch_empty = torch.empty

    def empty_without_pin_memory(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch_empty(*args, **kwargs)

    monkeypatch.setattr(megatron_utils.torch, "empty", empty_without_pin_memory)
    monkeypatch.setattr(megatron_utils, "_broadcast_offloaded_parameter", lambda tensor, unused_group: None)
    buffer = SimpleNamespace(param_data=torch.arange(12, dtype=torch.float32))

    megatron_utils._offload_ddp_param_buffer(buffer, group)
    cpu_data = buffer.param_data.cpu_data
    megatron_utils._load_ddp_param_buffer(buffer, group)
    buffer.param_data.add_(1)
    expected = buffer.param_data.clone()
    megatron_utils._offload_ddp_param_buffer(buffer, group)

    assert buffer.param_data.cpu_data is cpu_data
    torch.testing.assert_close(cpu_data, expected)


def test_deduplicated_buffer_requires_its_group_on_reload():
    buffer = SimpleNamespace(param_data=torch.ones(4))
    megatron_utils._offload_ddp_param_buffer(buffer, _non_owner_group())

    with pytest.raises(RuntimeError, match="missing parameter offload group"):
        megatron_utils._load_ddp_param_buffer(buffer, None)

    assert buffer.param_data.storage().size() == 0


def test_repeated_default_offload_validates_cpu_backing_size():
    param_data = torch.ones(4)
    param_data.cpu_data = torch.ones(3)
    buffer = SimpleNamespace(param_data=param_data, param_data_size=4)
    param_data.storage().resize_(0)

    with pytest.raises(AssertionError):
        megatron_utils._offload_ddp_param_buffer(buffer, None)


def test_expert_parameter_uses_expert_replica_group():
    dense_group = _non_owner_group()
    expert_group = _non_owner_group()
    groups = megatron_utils.MegatronParamOffloadGroups(dense=dense_group, expert=expert_group)
    dense = torch.nn.Parameter(torch.ones(1))
    expert = torch.nn.Parameter(torch.ones(1))
    expert.allreduce = False

    assert megatron_utils._parameter_offload_group(dense, groups) is dense_group
    assert megatron_utils._parameter_offload_group(expert, groups) is expert_group
    assert megatron_utils._parameter_offload_group(dense, None) is None


def test_snapshot_skips_non_owner_buffer_without_cpu_copy(monkeypatch):
    expected = torch.arange(12, dtype=torch.float32)
    buffer = SimpleNamespace(param_data=expected.clone())
    model = SimpleNamespace(buffers=[buffer], expert_parallel_buffers=[])
    monkeypatch.setattr(megatron_utils, "DDP", SimpleNamespace)
    group = _non_owner_group()
    groups = megatron_utils.MegatronParamOffloadGroups(dense=group, expert=None)

    megatron_utils._offload_ddp_param_buffer(buffer, group)
    state = megatron_utils.copy_megatron_model_to_cpu([model], param_offload_groups=groups)

    assert state["model_chunk_0"]["buffer_states"][0][0]["param_data"] is None
    with pytest.raises(ValueError, match="must match the saved Megatron CPU state"):
        megatron_utils.restore_megatron_model_from_cpu([model], state)
    monkeypatch.setattr(
        megatron_utils,
        "_broadcast_offloaded_parameter",
        lambda tensor, unused_group: tensor.copy_(expected),
    )
    megatron_utils._load_ddp_param_buffer(buffer, group)
    buffer.param_data.zero_()
    megatron_utils.restore_megatron_model_from_cpu([model], state, param_offload_groups=groups)
    torch.testing.assert_close(buffer.param_data, expected)


def test_default_snapshot_omits_deduplication_marker():
    state = megatron_utils.copy_megatron_model_to_cpu([torch.nn.Module()])

    assert "deduplicated" not in state


def test_non_ddp_snapshot_restores_loaded_non_owner_from_owner(monkeypatch):
    expected = torch.arange(6, dtype=torch.float32)
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(expected.clone()))
    group = _non_owner_group()
    groups = megatron_utils.MegatronParamOffloadGroups(dense=group, expert=None)

    megatron_utils._offload_replicated_parameter(model.weight, group)
    state = megatron_utils.copy_megatron_model_to_cpu([model], param_offload_groups=groups)
    assert state["model_chunk_0"]["model_state"]["weight"]["data"] is None

    monkeypatch.setattr(
        megatron_utils,
        "_broadcast_offloaded_parameter",
        lambda tensor, unused_group: tensor.copy_(expected),
    )
    megatron_utils._load_replicated_parameter(model.weight, group, "cpu")
    model.weight.zero_()
    megatron_utils.restore_megatron_model_from_cpu([model], state, param_offload_groups=groups)
    torch.testing.assert_close(model.weight, expected)
