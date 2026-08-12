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

import inspect
from types import MethodType, SimpleNamespace

import pytest
import torch
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import Float16Module

from verl.models.mcore import model_forward_fused as mff


def _new_uninitialized_model(model_cls=GPTModel):
    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    return model


def test_mcore_gpt_forward_has_native_output_processor_contract():
    parameters = inspect.signature(GPTModel.forward).parameters
    assert {"output_processor", "output_processor_context"}.issubset(parameters)


def test_native_hook_selection_preserves_forward_and_is_idempotent(monkeypatch):
    model = _new_uninitialized_model()
    original_forward = model.forward.__func__
    signature_calls = 0
    original_signature = inspect.signature

    def counted_signature(callable_):
        nonlocal signature_calls
        signature_calls += 1
        return original_signature(callable_)

    monkeypatch.setattr(mff.inspect, "signature", counted_signature)

    mff.patch_fused_forward(model)
    mff.patch_fused_forward(model)
    mff.unpatch_fused_forward(model)
    mff.unpatch_fused_forward(model)
    mff.patch_fused_forward(model)

    assert signature_calls == 1
    assert getattr(model, mff._FUSED_FORWARD_MODE_ATTR) == mff._HOOK_MODE
    assert model.forward.__func__ is original_forward
    assert not hasattr(model, "forward_backup")


def test_legacy_fallback_patch_and_unpatch_are_idempotent():
    class LegacyGPTModel(GPTModel):
        def forward(self, input_ids=None):
            return input_ids

    model = _new_uninitialized_model(LegacyGPTModel)
    original_forward = model.forward.__func__

    mff.patch_fused_forward(model)
    backup = model.forward_backup
    mff.patch_fused_forward(model)

    assert getattr(model, mff._FUSED_FORWARD_MODE_ATTR) == mff._LEGACY_MODE
    assert model.forward.__func__ is mff._fused_GPTModel_forward
    assert model.forward_backup is backup

    mff.unpatch_fused_forward(model)
    mff.unpatch_fused_forward(model)
    assert model.forward.__func__ is original_forward
    assert not hasattr(model, "forward_backup")

    mff.patch_fused_forward(model)
    assert model.forward.__func__ is mff._fused_GPTModel_forward
    mff.unpatch_fused_forward(model)
    assert model.forward.__func__ is original_forward


def test_engine_hook_context_and_current_thd_arguments(monkeypatch):
    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, 4]])
    preprocess_calls = []

    def fake_preprocess(value, **kwargs):
        preprocess_calls.append(kwargs)
        return value, "packed", None

    monkeypatch.setattr(mff, "preprocess_thd_engine", fake_preprocess)

    class Model:
        pre_process = True
        post_process = False
        config = SimpleNamespace(
            fp8="hybrid",
            csa_window_size=64,
            experimental_attention_variant="dsv4_hybrid",
        )

        def __init__(self, mode):
            setattr(self, mff._FUSED_FORWARD_MODE_ATTR, mode)
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(log_probs=torch.tensor([1.0]), entropy=torch.tensor([2.0]))

    hook_model = Model(mff._HOOK_MODE)
    output = mff.fused_forward_model_engine()(hook_model, input_ids, labels, {}, 0.7, True, 0, "dualpipev")

    hook_kwargs = hook_model.calls[-1]
    assert output.log_probs.tolist() == [1.0]
    assert "temperature" not in hook_kwargs
    assert hook_kwargs["output_processor"] is mff.fused_output_processor
    assert hook_kwargs["output_processor_context"].temperature == pytest.approx(0.7)
    assert preprocess_calls == [
        {
            "pre_process": True,
            "use_fp8_padding": True,
            "min_local_rows": 64,
            "pad_to_length_bucket": None,
            "cp_layout": "dualpipev",
            "local_cp_size": None,
        },
        {
            "pre_process": True,
            "need_roll": True,
            "use_fp8_padding": True,
            "min_local_rows": 64,
            "pad_to_length_bucket": None,
            "cp_layout": "dualpipev",
            "local_cp_size": None,
        },
    ]

    legacy_model = Model(mff._LEGACY_MODE)
    mff.fused_forward_model_engine()(legacy_model, input_ids, labels, {}, 0.5, True, 0)
    legacy_kwargs = legacy_model.calls[-1]
    assert legacy_kwargs["temperature"] == pytest.approx(0.5)
    assert "output_processor" not in legacy_kwargs
    assert "output_processor_context" not in legacy_kwargs


class _OutputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 3))


class _Decoder(torch.nn.Module):
    def forward(self, *, hidden_states, **_kwargs):
        return hidden_states


def test_megatron_bridge_wrapper_chain_reaches_native_hook(monkeypatch):
    """Exercise the Float16Module -> DDP stack built by Megatron Bridge."""
    model = _new_uninitialized_model()
    model.config = SimpleNamespace(
        fine_grained_activation_offloading=False,
        moe_paged_stash=False,
        mtp_num_layers=0,
        use_mup=False,
        sequence_parallel=False,
    )
    model.share_embeddings_and_output_weights = False
    model.mtp_process = False
    model.post_process = True
    model.output_layer = _OutputLayer()
    model.decoder = _Decoder()
    model.pg_collection = SimpleNamespace(tp=None, cp=None)
    hidden_states = torch.ones(2, 1, 3)

    def preprocess(_self, **_kwargs):
        return hidden_states, None, None, None, None, None

    model._preprocess = MethodType(preprocess, model)

    mixed_precision_wrapper = object.__new__(Float16Module)
    torch.nn.Module.__init__(mixed_precision_wrapper)
    mixed_precision_wrapper.module = model
    mixed_precision_wrapper.pg_collection = SimpleNamespace(pp=SimpleNamespace(rank=lambda: 0, size=lambda: 1))
    mixed_precision_wrapper.vp_stage = None
    mixed_precision_wrapper.vp_size = None
    mixed_precision_wrapper.float16_convertor = lambda value: value

    ddp_wrapper = object.__new__(DistributedDataParallel)
    torch.nn.Module.__init__(ddp_wrapper)
    ddp_wrapper.module = mixed_precision_wrapper

    seen = {}

    def fake_linear_cross_entropy(hidden, weight, labels, temperature, reduction, group):
        seen.update(
            hidden=hidden,
            weight=weight,
            labels=labels,
            temperature=temperature,
            reduction=reduction,
            group=group,
        )
        return torch.ones(2), torch.ones(2) * 2

    monkeypatch.setattr(mff, "linear_cross_entropy", fake_linear_cross_entropy)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: None)

    original_forward = model.forward.__func__
    mff.patch_fused_forward(ddp_wrapper)
    output = ddp_wrapper(
        input_ids=torch.tensor([[1, 2]]),
        position_ids=torch.tensor([[0, 1]]),
        attention_mask=None,
        labels=torch.tensor([1, 2]),
        output_processor=mff.fused_output_processor,
        output_processor_context=mff.FusedOutputProcessorContext(temperature=0.7),
        fp32_output=False,
    )

    assert model.forward.__func__ is original_forward
    assert not hasattr(model, "forward_backup")
    assert output.log_probs.tolist() == [1.0, 1.0]
    assert output.entropy.tolist() == [2.0, 2.0]
    assert seen["temperature"] == pytest.approx(0.7)
    assert seen["weight"] is model.output_layer.weight


def test_output_processor_preserves_config_logger_payload(monkeypatch):
    input_ids = object()
    position_ids = object()
    attention_mask = object()
    decoder_input = object()
    log_probs = torch.tensor([1.0])
    entropy = torch.tensor([2.0])
    config = SimpleNamespace(sequence_parallel=False)
    logged = {}

    def fake_linear_cross_entropy(*_args, **_kwargs):
        return log_probs, entropy

    def fake_log_config_to_disk(config_arg, payload, prefix):
        logged.update(config=config_arg, payload=payload, prefix=prefix)

    monkeypatch.setattr(mff, "linear_cross_entropy", fake_linear_cross_entropy)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(mff, "has_config_logger_enabled", lambda config_arg: config_arg is config)
    monkeypatch.setattr(mff, "log_config_to_disk", fake_log_config_to_disk)

    mff.fused_output_processor(
        hidden_states=torch.tensor([[1.0]]),
        output_layer=SimpleNamespace(weight=torch.tensor([[1.0]])),
        output_weight=None,
        labels=torch.tensor([0]),
        context=mff.FusedOutputProcessorContext(temperature=1.0),
        config=config,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        decoder_input=decoder_input,
    )

    assert list(logged["payload"]) == [
        "input_ids",
        "position_ids",
        "attention_mask",
        "decoder_input",
        "logprobs",
        "entropy",
    ]
    assert logged["payload"]["input_ids"] is input_ids
    assert logged["payload"]["position_ids"] is position_ids
    assert logged["payload"]["attention_mask"] is attention_mask
    assert logged["payload"]["decoder_input"] is decoder_input
    assert logged["payload"]["logprobs"] is log_probs
    assert logged["payload"]["entropy"] is entropy
    assert logged["config"] is config
    assert logged["prefix"] == "input_and_logits"


@pytest.mark.parametrize(
    ("sequence_parallel", "use_tied_weight"),
    [(True, True), (False, False)],
)
def test_output_processor_gathers_before_kernel_and_resolves_weight(monkeypatch, sequence_parallel, use_tied_weight):
    hidden_states = torch.tensor([[1.0]])
    gathered_hidden_states = torch.tensor([[3.0]])
    tied_weight = torch.tensor([[5.0]])
    output_layer_weight = torch.tensor([[7.0]])
    labels = torch.tensor([0])
    events = []
    seen = {}

    def fake_gather(value):
        events.append("gather")
        assert value is hidden_states
        return gathered_hidden_states

    def fake_linear_cross_entropy(hidden, weight, labels_arg, temperature, reduction, group):
        events.append("linear_cross_entropy")
        seen.update(hidden=hidden, weight=weight, labels=labels_arg, temperature=temperature)
        return torch.tensor([11.0]), torch.tensor([13.0])

    monkeypatch.setattr(mff, "gather_from_sequence_parallel_region", fake_gather)
    monkeypatch.setattr(mff, "linear_cross_entropy", fake_linear_cross_entropy)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: None)

    output_weight = tied_weight if use_tied_weight else None
    output = mff.fused_output_processor(
        hidden_states=hidden_states,
        output_layer=SimpleNamespace(weight=output_layer_weight),
        output_weight=output_weight,
        labels=labels,
        context=mff.FusedOutputProcessorContext(temperature=0.9),
        config=SimpleNamespace(sequence_parallel=sequence_parallel),
    )

    expected_hidden = gathered_hidden_states if sequence_parallel else hidden_states
    expected_weight = tied_weight if use_tied_weight else output_layer_weight
    assert events == (["gather"] if sequence_parallel else []) + ["linear_cross_entropy"]
    assert seen["hidden"] is expected_hidden
    assert seen["weight"] is expected_weight
    assert seen["labels"] is labels
    assert seen["temperature"] == pytest.approx(0.9)
    assert output.log_probs.tolist() == [11.0]
    assert output.entropy.tolist() == [13.0]
