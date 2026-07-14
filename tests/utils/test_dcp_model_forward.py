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
from tensordict import TensorDict

import verl.models.mcore as mcore_module
import verl.models.mcore.model_forward as model_forward_module
import verl.models.mcore.model_forward_fused as fused_forward_module
import verl.workers.engine.megatron.transformer_impl as transformer_module
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config.engine import McoreEngineConfig, get_mcore_parallel_topology
from verl.workers.engine.megatron.transformer_impl import (
    MegatronEngine,
    MegatronEngineWithLMHead,
    _resolve_dcp_transformer_overrides,
    _validate_dcp_model_features,
)


class _IntermediateValueStage:
    pre_process = False
    post_process = False
    config = SimpleNamespace(fp8=None)

    def __init__(self, activation):
        self.activation = activation

    def __call__(self, **_kwargs):
        return self.activation


def test_dcp_value_intermediate_pipeline_stage_returns_activation_tensor(monkeypatch):
    activation = torch.randn(1, 4, 8)
    model = _IntermediateValueStage(activation)
    input_ids = torch.nested.nested_tensor([torch.arange(4)], layout=torch.jagged)

    monkeypatch.setattr(
        model_forward_module,
        "preprocess_thd_engine",
        lambda input_ids, **_kwargs: (input_ids, object(), None),
    )
    monkeypatch.setattr(
        model_forward_module,
        "postprocess_thd_engine_local",
        lambda output, *_args, **_kwargs: output,
    )

    output = model_forward_module.gptmodel_forward_model_engine(
        model=model,
        input_ids=input_ids,
        multi_modal_inputs={},
        value_model=True,
        local_cp_size=2,
        dcp_local_output_only=True,
        dcp_compact_output_only=True,
        return_dcp_local_token_mask=True,
    )

    assert isinstance(output, torch.Tensor)
    assert output is activation


@pytest.mark.parametrize("max_seqlen", [None, 0, -1, 1.5, True])
def test_dcp_config_requires_positive_integer_sequence_limit(max_seqlen):
    with pytest.raises(ValueError, match="max_seqlen_per_dp_cp_rank must be a positive integer"):
        McoreEngineConfig(dynamic_context_parallel=True, max_seqlen_per_dp_cp_rank=max_seqlen)


def test_dcp_config_accepts_positive_integer_sequence_limit():
    config = McoreEngineConfig(dynamic_context_parallel=True, max_seqlen_per_dp_cp_rank=4096)

    assert config.max_seqlen_per_dp_cp_rank == 4096


def _dcp_engine_config(**overrides):
    values = {
        "dynamic_context_parallel": True,
        "context_parallel_size": 4,
        "max_seqlen_per_dp_cp_rank": 4096,
        "override_ddp_config": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dcp_transformer_overrides_keep_static_cp_and_enable_per_token_loss():
    engine_config = _dcp_engine_config()

    overrides = _resolve_dcp_transformer_overrides(engine_config, {"attention_softmax_in_fp32": True})

    assert overrides == {
        "attention_softmax_in_fp32": True,
        "calculate_per_token_loss": True,
        "context_parallel_size": 4,
        "dynamic_context_parallel": False,
        "max_seqlen_per_dp_cp_rank": 4096,
    }


def test_dcp_validates_resolved_bridge_transformer_config():
    engine_config = _dcp_engine_config()
    resolved = SimpleNamespace(
        calculate_per_token_loss=True,
        context_parallel_size=4,
        dynamic_context_parallel=False,
        max_seqlen_per_dp_cp_rank=4096,
        moe_z_loss_coeff=0.0,
    )

    transformer_module._validate_resolved_dcp_transformer_config(engine_config, resolved)

    resolved.calculate_per_token_loss = False
    with pytest.raises(ValueError, match="calculate_per_token_loss"):
        transformer_module._validate_resolved_dcp_transformer_config(engine_config, resolved)


def test_dcp_rejects_resolved_bridge_moe_z_loss():
    engine_config = _dcp_engine_config()
    resolved = SimpleNamespace(
        calculate_per_token_loss=True,
        context_parallel_size=4,
        dynamic_context_parallel=False,
        max_seqlen_per_dp_cp_rank=4096,
        moe_z_loss_coeff=1e-3,
    )

    with pytest.raises(NotImplementedError, match="resolved Megatron-Core"):
        transformer_module._validate_resolved_dcp_transformer_config(engine_config, resolved)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"context_parallel_size": 8}, "context_parallel_size to match"),
        ({"calculate_per_token_loss": False}, "calculate_per_token_loss=True"),
    ],
)
def test_dcp_transformer_overrides_reject_incompatible_normalization_or_topology(overrides, match):
    with pytest.raises(ValueError, match=match):
        _resolve_dcp_transformer_overrides(_dcp_engine_config(), overrides)


def test_dcp_ddp_overrides_disable_collective_average():
    engine = object.__new__(MegatronEngine)
    engine.engine_config = _dcp_engine_config(override_ddp_config={"overlap_grad_reduce": True})
    engine.optimizer_config = None

    assert engine._resolve_override_ddp_config() == {
        "average_in_collective": False,
        "overlap_grad_reduce": True,
    }


def test_dcp_ddp_overrides_reject_collective_average():
    engine = object.__new__(MegatronEngine)
    engine.engine_config = _dcp_engine_config(override_ddp_config={"average_in_collective": True})
    engine.optimizer_config = None

    with pytest.raises(ValueError, match="average_in_collective=False"):
        engine._resolve_override_ddp_config()


def test_dcp_rejects_mtp_training_until_global_normalization_is_supported():
    model_config = SimpleNamespace(mtp=SimpleNamespace(enable=True, enable_train=True))
    engine_config = SimpleNamespace(dynamic_context_parallel=True)

    with pytest.raises(NotImplementedError, match="does not yet support MTP training"):
        _validate_dcp_model_features(model_config, engine_config)


def test_dcp_rejects_moe_z_loss_with_rank_local_normalization():
    model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False, enable_train=False))
    engine_config = SimpleNamespace(
        dynamic_context_parallel=True,
        override_transformer_config={"moe_z_loss_coeff": 1e-3},
    )

    with pytest.raises(NotImplementedError, match="does not yet support moe_z_loss_coeff"):
        _validate_dcp_model_features(model_config, engine_config)


@pytest.mark.parametrize("moe_z_loss_coeff", [None, 0, 0.0])
def test_dcp_allows_disabled_moe_z_loss(moe_z_loss_coeff):
    model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False, enable_train=False))
    engine_config = SimpleNamespace(
        dynamic_context_parallel=True,
        override_transformer_config={"moe_z_loss_coeff": moe_z_loss_coeff},
    )

    _validate_dcp_model_features(model_config, engine_config)


def test_topk_distillation_falls_back_from_fused_forward(monkeypatch):
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.engine_config = SimpleNamespace(dynamic_context_parallel=False, use_remove_padding=True)
    engine.model_config = SimpleNamespace(
        hf_config=object(),
        tokenizer=SimpleNamespace(pad_token_id=0),
        mtp=SimpleNamespace(enable=False, enable_train=False),
    )
    engine.tf_config = SimpleNamespace()
    input_ids = torch.nested.nested_tensor([torch.arange(4)], layout=torch.jagged)
    batch = TensorDict({"input_ids": input_ids}, batch_size=[1])
    tu.assign_non_tensor(
        batch,
        use_fused_kernels=True,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=True,
        distillation_only=True,
        pad_mode=DatasetPadMode.NO_PADDING,
    )
    engine.prepare_model_inputs = lambda _batch: {
        "input_ids": input_ids,
        "attention_mask": None,
        "multi_modal_inputs": {},
        "loss_mask": None,
    }

    marker = {"distillation_losses": object()}
    fallback_called = False

    def fake_forward_fn(*_args, **_kwargs):
        nonlocal fallback_called
        fallback_called = True
        return marker

    monkeypatch.setattr(transformer_module, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(transformer_module, "unwrap_model", lambda model: model)
    monkeypatch.setattr(transformer_module.RouterReplayHelper, "is_replay_backward_action", lambda *_args: False)
    monkeypatch.setattr(transformer_module.RouterReplayHelper, "is_replay_forward_action", lambda *_args: False)
    monkeypatch.setattr(transformer_module.RouterReplayHelper, "is_r2_record_action", lambda *_args: False)
    monkeypatch.setattr(mcore_module, "get_mcore_engine_forward_fn", lambda _config: fake_forward_fn)
    monkeypatch.setattr(
        mcore_module,
        "get_mcore_forward_fused_model_engine_fn",
        lambda _config: pytest.fail("top-k distillation must not use the fused forward"),
    )

    output, _postprocess = engine.forward_step(
        iter([batch]),
        SimpleNamespace(),
        logits_processor_func=object(),
        postprocess_micro_batch_func=lambda *_args, **_kwargs: None,
        forward_only=True,
    )

    assert fallback_called
    assert output is marker


def test_non_fused_wrapper_bypasses_installed_fused_forward():
    class PatchedModel(torch.nn.Module):
        def forward(self, value):
            return value + 1

    model = PatchedModel()
    model.forward_backup = model.forward

    def fused_forward(_self, **_kwargs):
        raise AssertionError("the installed fused forward must be bypassed")

    model.forward = fused_forward.__get__(model, PatchedModel)
    installed_fused_forward = model.forward

    output = model_forward_module._call_model_with_unfused_forward(model, value=torch.tensor(2))

    torch.testing.assert_close(output, torch.tensor(3))
    assert model.forward == installed_fused_forward


def test_fused_forward_patch_unpatch_repatch_refreshes_backup(monkeypatch):
    class FakeModel:
        def original_forward(self):
            return "original"

        def replacement_forward(self):
            return "replacement"

    model = FakeModel()
    model.forward = model.original_forward
    monkeypatch.setattr(fused_forward_module, "_get_patching_model", lambda _model: model)
    monkeypatch.setattr(fused_forward_module.mcore, "__version__", "0.14.0")

    fused_forward_module.patch_fused_forward(model)
    fused_forward_module.unpatch_fused_forward(model)
    assert not hasattr(model, "forward_backup")
    assert model.forward() == "original"

    model.forward = model.replacement_forward
    fused_forward_module.patch_fused_forward(model)
    fused_forward_module.unpatch_fused_forward(model)
    assert not hasattr(model, "forward_backup")
    assert model.forward() == "replacement"


def test_mcore_topology_normalizes_default_expert_tensor_parallel_size():
    config = SimpleNamespace(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=3,
        context_parallel_size=2,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=None,
    )

    assert get_mcore_parallel_topology(config) == {
        "tensor": 4,
        "pipeline": 2,
        "virtual_pipeline": 3,
        "context": 2,
        "expert": 8,
        "expert_tensor": 4,
    }


@pytest.mark.parametrize(
    ("getter_name", "mismatched_value"),
    [
        ("get_virtual_pipeline_model_parallel_world_size", 3),
        ("get_expert_model_parallel_world_size", 2),
        ("get_expert_tensor_parallel_world_size", 2),
    ],
)
def test_dcp_rejects_reusing_incompatible_initialized_megatron_topology(monkeypatch, getter_name, mismatched_value):
    engine = object.__new__(MegatronEngine)
    engine.engine_config = SimpleNamespace(
        dynamic_context_parallel=True,
        use_remove_padding=True,
        max_seqlen_per_dp_cp_rank=1024,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        context_parallel_size=2,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=None,
    )
    actual = {
        "get_tensor_model_parallel_world_size": 1,
        "get_pipeline_model_parallel_world_size": 2,
        "get_virtual_pipeline_model_parallel_world_size": 2,
        "get_context_parallel_world_size": 2,
        "get_expert_model_parallel_world_size": 1,
        "get_expert_tensor_parallel_world_size": 1,
    }
    actual[getter_name] = mismatched_value
    monkeypatch.setattr(transformer_module.torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(transformer_module.mpu, "is_initialized", lambda: True)
    for name, value in actual.items():
        monkeypatch.setattr(transformer_module.mpu, name, lambda value=value: value)

    with pytest.raises(ValueError, match="TP/PP/VPP/CP/EP/ETP topology"):
        engine._init_device_mesh()
