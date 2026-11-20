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

"""CPU tests for the router replay startup guards.

The guards turn unsupported router replay setups into explicit failures
instead of silent route corruption:

* ``_validate_router_replay_setup`` rejects ``moe_permute_fusion`` /
  ``moe_router_fusion`` / ``moe_router_topk == 1`` and warns for
  dispatchers other than ``alltoall``.
* ``apply_router_replay_patch`` raises a clear error when Megatron-Core's
  MoE helpers failed to import, instead of leaving helper symbols unbound.
"""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils.megatron import router_replay_patch as replay_patch
from verl.workers.engine.megatron import transformer_impl


def _configs(mode="R3", **overrides):
    tf_values = {
        "moe_permute_fusion": False,
        "moe_router_fusion": False,
        "moe_token_dispatcher_type": "alltoall",
        "moe_router_topk": 8,
        "num_layers": 4,
    }
    tf_values.update(overrides)
    engine_config = SimpleNamespace(router_replay=SimpleNamespace(mode=mode))
    return engine_config, SimpleNamespace(**tf_values)


def test_disabled_mode_skips_all_checks():
    """Disabled mode must skip every replay setup check."""
    engine_config, tf_config = _configs(
        mode="disabled", moe_permute_fusion=True, moe_router_fusion=True, moe_router_topk=1
    )

    transformer_impl._validate_router_replay_setup(engine_config, tf_config)


@pytest.mark.parametrize(
    "override",
    [
        {"moe_permute_fusion": True},
        {"moe_router_fusion": True},
        {"moe_router_topk": 1},
    ],
)
def test_rejects_unsupported_setup(override):
    """Unsupported fusion/topk combinations must be rejected."""
    engine_config, tf_config = _configs(**override)

    with pytest.raises(ValueError, match="router_replay"):
        transformer_impl._validate_router_replay_setup(engine_config, tf_config)


def test_warns_for_non_alltoall_dispatcher(monkeypatch):
    """A non-alltoall dispatcher is accepted but emits a warning."""
    engine_config, tf_config = _configs(moe_token_dispatcher_type="flex")
    messages = []

    monkeypatch.setattr(transformer_impl.logger, "warning", lambda msg, *args: messages.append(msg % args))

    transformer_impl._validate_router_replay_setup(engine_config, tf_config)

    assert any("alltoall" in message for message in messages)


def test_patch_rejects_degraded_import(monkeypatch):
    """Enabling replay with missing MoE helpers must fail clearly."""
    monkeypatch.setattr(replay_patch, "_REPLAY_IMPORT_ERROR", ImportError("missing moe_utils"))

    with pytest.raises(RuntimeError, match="MoE helpers are unavailable"):
        replay_patch.apply_router_replay_patch()


def test_clear_replay_state_after_error_keeps_action_cleanup_on_index_failure(monkeypatch):
    """A failing index clear must not skip action cleanup (and must not escape)."""
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.module = object()
    engine.mini_layer_topk_idx_list = [object()]
    cleared = []

    def fail(_model):
        raise RuntimeError("index clear failed")

    monkeypatch.setattr(transformer_impl.logger, "exception", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "clear_model_router_replay_indices", fail)
    monkeypatch.setattr(transformer_impl, "clear_model_router_replay_action", lambda _model: cleared.append(True))

    engine._clear_replay_state_after_error()

    assert engine.mini_layer_topk_idx_list == []
    assert cleared == [True]


def test_forward_backward_batch_clears_state_on_postprocess_error(monkeypatch):
    """A postprocessing failure must still clear replay state before propagating."""
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
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.tf_config = None
    engine.module = []
    engine.get_data_parallel_group = lambda: None
    engine.get_data_parallel_size = lambda: 1

    data = TensorDict({"loss_mask": torch.ones(4)}, batch_size=[4])
    transformer_impl.tu.assign_non_tensor_data(data, "enable_routing_replay", True)

    monkeypatch.setattr(transformer_impl, "_check_dcp_unsupported_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(transformer_impl.torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(transformer_impl, "prepare_micro_batches", lambda **kwargs: ([], None))
    monkeypatch.setattr(transformer_impl, "make_batch_generator", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "get_forward_backward_func", lambda: (lambda **kwargs: []))
    monkeypatch.setattr(transformer_impl.mpu, "is_pipeline_last_stage", lambda ignore_virtual=True: True)

    def boom(**kwargs):
        raise RuntimeError("postprocess boom")

    monkeypatch.setattr(transformer_impl, "postprocess_batch_func", boom)

    cleared = []
    monkeypatch.setattr(engine, "_clear_replay_state_after_error", lambda: cleared.append(True))

    with pytest.raises(RuntimeError, match="postprocess boom"):
        engine.forward_backward_batch(data, loss_function=None)

    assert cleared == [True]
