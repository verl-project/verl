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

"""CPU tests for the router replay batch-flag guard."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.workers.engine.megatron import transformer_impl
from verl.workers.engine.megatron.transformer_impl import _validate_router_replay_batch_flag


def test_guard_allows_disabled_engine():
    _validate_router_replay_batch_flag(False, False)


def test_guard_allows_enabled_engine_with_flag():
    _validate_router_replay_batch_flag(True, True)


def test_guard_rejects_missing_batch_flag():
    with pytest.raises(RuntimeError, match="missing the 'enable_routing_replay' flag"):
        _validate_router_replay_batch_flag(True, False)


def test_forward_backward_batch_rejects_missing_batch_flag(monkeypatch):
    """The guard is wired into forward_backward_batch, not only callable directly."""
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.enable_routing_replay = True
    engine.engine_config = SimpleNamespace(
        context_parallel_size=1,
        dynamic_context_parallel=False,
        pad_bshd_to_minibatch_max=False,
        use_remove_padding=True,
        router_replay=SimpleNamespace(mode="R3"),
        use_fused_kernels=False,
    )
    engine.model_config = object()
    engine.tf_config = None
    engine.get_data_parallel_group = lambda: None
    engine.get_data_parallel_size = lambda: 1

    data = TensorDict({"loss_mask": torch.ones(4)}, batch_size=[4])

    monkeypatch.setattr(transformer_impl, "_check_dcp_unsupported_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(transformer_impl.torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(transformer_impl, "prepare_micro_batches", lambda **kwargs: ([], None))

    with pytest.raises(RuntimeError, match="missing the 'enable_routing_replay' flag"):
        engine.forward_backward_batch(data, loss_function=None)
