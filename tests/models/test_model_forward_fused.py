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
from verl.models.mcore import registry


def _new_uninitialized_model(model_cls=GPTModel):
    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    return model


def _skip_without_hybrid_hook():
    if mff.HybridModel is None:
        pytest.skip("Megatron does not provide HybridModel.")
    if not mff._supports_output_processor_hook(_new_uninitialized_model(mff.HybridModel)):
        pytest.skip("Installed HybridModel has no output-processor hook.")


def test_mcore_gpt_forward_has_native_output_processor_contract():
    parameters = inspect.signature(GPTModel.forward).parameters
    assert {"output_processor", "output_processor_context"}.issubset(parameters)


@pytest.mark.parametrize(
    "architecture,vision_model",
    [
        ("Glm5NextForConditionalGeneration", True),
        ("Qwen3VLForConditionalGeneration", True),
        ("LlamaForCausalLM", False),
    ],
)
def test_engine_registry_uses_supported_model_layout(monkeypatch, architecture, vision_model):
    config = SimpleNamespace(architectures=[architecture], vision_config=object())
    monkeypatch.setattr(registry, "gptmodel_forward_model_engine", lambda **kwargs: kwargs)
    assert registry.get_mcore_engine_forward_fn(config)()["vision_model"] is vision_model


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


def test_hybrid_wrapper_without_hook_uses_its_language_model_hook():
    """GLM-5.3-Flash: the wrapper forwards only Megatron-Core args, so the hook goes to its inner HybridModel."""
    _skip_without_hybrid_hook()
    wrapper = torch.nn.Module()
    wrapper.language_model = _new_uninitialized_model(mff.HybridModel)

    assert mff.patch_fused_forward(wrapper) is True
    mff.unpatch_fused_forward(wrapper)

    assert getattr(wrapper, mff._FUSED_FORWARD_MODE_ATTR) == mff._LANGUAGE_MODEL_HOOK_MODE
    assert not hasattr(wrapper, "forward_backup")
    assert "forward" not in vars(wrapper)


def test_engine_disables_fused_kernels_when_a_chunk_is_unsupported(monkeypatch):
    from verl.workers.engine.megatron import transformer_impl

    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = SimpleNamespace(use_fused_kernels=True, use_remove_padding=True)
    engine.model_config = SimpleNamespace(use_liger=False, mtp=SimpleNamespace(enable=False))
    engine.is_value_model = False
    engine.module = [object(), object()]
    unpatched = []
    monkeypatch.setattr(mff, "patch_fused_forward", lambda model, config, **kwargs: model is engine.module[0])
    monkeypatch.setattr(mff, "unpatch_fused_forward", unpatched.append)

    engine._maybe_enable_fused_kernels()

    assert engine.engine_config.use_fused_kernels is False
    assert unpatched == engine.module


def test_engine_hands_hook_and_mtp_choice_to_language_model_of_full_stream_vlm(monkeypatch):
    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, 4]])
    full_ids = torch.tensor([[1, 2, 3, 0]])
    full_labels = torch.tensor([[2, 3, 0, 1]])
    full_stream_calls = []

    monkeypatch.setattr(mff, "preprocess_thd_engine", lambda value, **kwargs: (value, "packed", None))
    monkeypatch.setattr(mff, "takes_full_packed_thd_vlm_inputs", lambda model: True)
    monkeypatch.setattr(
        mff,
        "preprocess_vlm_thd_engine",
        lambda model, ids, rmpad, params, position_ids, pad, **kw: (full_ids, None, None),
    )

    def fake_full_stream(values, packed_seq_params, need_roll=False):
        full_stream_calls.append((values, packed_seq_params, need_roll))
        return full_labels

    monkeypatch.setattr(mff, "preprocess_thd_full_stream", fake_full_stream)

    class LanguageModel:
        def __init__(self):
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(log_probs=torch.tensor([1.0]), entropy=torch.tensor([2.0]))

    class Wrapper:
        pre_process = True
        post_process = False
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            self.language_model = LanguageModel()
            self.calls = []
            setattr(self, mff._FUSED_FORWARD_MODE_ATTR, mff._LANGUAGE_MODEL_HOOK_MODE)

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return self.language_model.forward(input_ids=kwargs["input_ids"], labels=kwargs["labels"])

    model = Wrapper()
    output = mff.fused_forward_model_engine(vision_model=True)(model, input_ids, labels, {}, 0.7, True, 0)

    assert output.log_probs.tolist() == [1.0]
    wrapper_kwargs = model.calls[-1]
    assert wrapper_kwargs["input_ids"] is full_ids
    assert wrapper_kwargs["labels"] is full_labels
    assert wrapper_kwargs["attention_mask"] is None
    assert not {"output_processor", "output_processor_context", "temperature"} & set(wrapper_kwargs)
    assert full_stream_calls == [(labels, "packed", True)]
    language_model_kwargs = model.language_model.calls[-1]
    assert language_model_kwargs["output_processor"] is mff.fused_output_processor
    assert language_model_kwargs["output_processor_context"].temperature == pytest.approx(0.7)
    assert language_model_kwargs["compute_mtp_loss"] is False
    assert "forward" not in vars(model.language_model)


@pytest.mark.parametrize("post_process", [True, False])
def test_full_stream_vlm_output_and_loss_mask_tuple_is_unpacked(monkeypatch, post_process):
    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, 4]])
    model_output = SimpleNamespace(log_probs=torch.tensor([[0.1, 0.2]]), entropy=torch.tensor([[0.3, 0.4]]))

    monkeypatch.setattr(mff, "preprocess_thd_engine", lambda value, **kwargs: (value, "packed", None))
    monkeypatch.setattr(mff, "takes_full_packed_thd_vlm_inputs", lambda model: True)
    monkeypatch.setattr(
        mff,
        "preprocess_vlm_thd_engine",
        lambda model, ids, rmpad, params, position_ids, pad, **kw: (ids, None, None),
    )
    monkeypatch.setattr(mff, "preprocess_thd_full_stream", lambda values, packed_seq_params, need_roll=False: values)
    monkeypatch.setattr(mff, "postprocess_thd_engine", lambda value, *args, **kwargs: value)

    class LanguageModel:
        def forward(self, **kwargs):
            return model_output

    class Wrapper:
        pre_process = True
        config = SimpleNamespace(fp8=None)

        def __init__(self):
            self.post_process = post_process
            self.language_model = LanguageModel()
            setattr(self, mff._FUSED_FORWARD_MODE_ATTR, mff._LANGUAGE_MODEL_HOOK_MODE)

        def __call__(self, **kwargs):
            # GLM-5.3-Flash returns its CP-sliced loss mask next to the output.
            return self.language_model.forward(**kwargs), torch.ones(2)

    output = mff.fused_forward_model_engine(vision_model=True)(Wrapper(), input_ids, labels, {}, 1.0, True, 0)

    if post_process:
        assert torch.equal(output["log_probs"], model_output.log_probs)
        assert torch.equal(output["entropy"], model_output.entropy)
    else:
        assert output is model_output


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
    setattr(hook_model, mff._FUSED_IMPL_BACKEND_ATTR, "liger")
    output = mff.fused_forward_model_engine()(hook_model, input_ids, labels, {}, 0.7, True, 0, "dualpipev")

    hook_kwargs = hook_model.calls[-1]
    assert output.log_probs.tolist() == [1.0]
    assert "temperature" not in hook_kwargs
    assert hook_kwargs["output_processor"] is mff.fused_output_processor
    assert hook_kwargs["output_processor_context"].temperature == pytest.approx(0.7)
    assert hook_kwargs["output_processor_context"].impl_backend == "liger"
    assert preprocess_calls == [
        {
            "pre_process": True,
            "quant_align_size": 16,
            "min_local_rows": 64,
            "pad_to_length_bucket": None,
            "cp_layout": "dualpipev",
            "local_cp_size": None,
        },
        {
            "pre_process": True,
            "need_roll": True,
            "quant_align_size": 16,
            "min_local_rows": 64,
            "pad_to_length_bucket": None,
            "cp_layout": "dualpipev",
            "local_cp_size": None,
        },
    ]

    legacy_model = Model(mff._LEGACY_MODE)
    setattr(legacy_model, mff._FUSED_IMPL_BACKEND_ATTR, "liger")
    mff.fused_forward_model_engine()(legacy_model, input_ids, labels, {}, 0.5, True, 0)
    legacy_kwargs = legacy_model.calls[-1]
    assert legacy_kwargs["temperature"] == pytest.approx(0.5)
    assert legacy_kwargs["impl_backend"] == "liger"
    assert "output_processor" not in legacy_kwargs
    assert "output_processor_context" not in legacy_kwargs


def test_engine_processes_tilelang_topk_before_thd_postprocess(monkeypatch):
    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, 4]])
    topk_labels = torch.tensor([[[5, 5], [6, 7], [8, 9]]])
    raw_topk = torch.randn(1, 3, 2, requires_grad=True)
    processor_inputs = []

    monkeypatch.setattr(mff, "preprocess_thd_engine", lambda value, **_kwargs: (value, "packed", None))
    monkeypatch.setattr(mff, "postprocess_thd_engine", lambda value, *_args, **_kwargs: value)

    class Model:
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None, csa_window_size=None, experimental_attention_variant=None)

        def __init__(self):
            setattr(self, mff._FUSED_FORWARD_MODE_ATTR, mff._HOOK_MODE)
            setattr(self, mff._FUSED_IMPL_BACKEND_ATTR, "tilelang")
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                log_probs=torch.zeros(1, 3),
                topk_log_probs=raw_topk,
                entropy=torch.ones(1, 3),
            )

    def auxiliary_processor(*, student_topk_log_probs):
        processor_inputs.append(student_topk_log_probs)
        return {"distillation_losses": student_topk_log_probs.sum(dim=-1)}

    model = Model()
    output = mff.fused_forward_model_engine()(
        model,
        input_ids,
        labels,
        {},
        1.0,
        False,
        0,
        topk_labels=topk_labels,
        auxiliary_logits_processor=auxiliary_processor,
    )

    assert model.kwargs["output_processor_context"].topk_labels is topk_labels
    assert processor_inputs == [raw_topk]
    assert set(output) == {"log_probs", "distillation_losses"}
    torch.testing.assert_close(output["distillation_losses"], raw_topk.sum(dim=-1))


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
        freeze_base_model_for_mtp=False,
        cuda_graph_impl="none",
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

    def fake_linear_cross_entropy(hidden, weight, labels, temperature, reduction, group, **options):
        seen.update(
            hidden=hidden,
            weight=weight,
            labels=labels,
            temperature=temperature,
            reduction=reduction,
            group=group,
            options=options,
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
        output_processor_context=mff.FusedOutputProcessorContext(
            temperature=0.7,
            impl_backend="liger",
        ),
        fp32_output=False,
    )

    assert model.forward.__func__ is original_forward
    assert not hasattr(model, "forward_backup")
    assert output.log_probs.tolist() == [1.0, 1.0]
    assert output.entropy.tolist() == [2.0, 2.0]
    assert seen["temperature"] == pytest.approx(0.7)
    assert seen["weight"] is model.output_layer.weight
    assert seen["options"] == {"impl_backend": "liger"}


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


def test_output_processor_aligns_mcore_sequence_batch_layout_for_topk(monkeypatch):
    hidden_states = torch.randn(3, 2, 4)
    labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
    topk_labels = torch.tensor(
        [
            [[7, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 14], [15, 16]],
        ]
    )
    seen = {}

    def fake_linear_cross_entropy(hidden, _weight, kernel_labels, *_args, **_kwargs):
        seen["hidden"] = hidden
        seen["labels"] = kernel_labels
        return kernel_labels.float(), torch.zeros(kernel_labels.shape[:-1])

    monkeypatch.setattr(mff, "linear_cross_entropy", fake_linear_cross_entropy)
    monkeypatch.setattr(mff, "copy_to_tensor_model_parallel_region", lambda value: value)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: None)

    output = mff.fused_output_processor(
        hidden_states=hidden_states,
        output_layer=SimpleNamespace(weight=torch.randn(17, 4)),
        output_weight=None,
        labels=labels,
        context=mff.FusedOutputProcessorContext(
            temperature=1.0,
            impl_backend="tilelang",
            topk_labels=topk_labels,
        ),
        config=SimpleNamespace(sequence_parallel=False),
    )

    expected_kernel_labels = torch.cat((labels.unsqueeze(-1), topk_labels), dim=-1).transpose(0, 1)
    assert seen["hidden"] is hidden_states
    assert torch.equal(seen["labels"], expected_kernel_labels)
    torch.testing.assert_close(output.log_probs, labels.float())
    torch.testing.assert_close(output.topk_log_probs, topk_labels.float())
    assert output.entropy.shape == labels.shape


@pytest.mark.parametrize(
    ("sequence_parallel", "use_tied_weight", "impl_backend", "tensor_parallel_output_grad"),
    [
        (True, True, "liger", False),
        (True, False, "triton", True),
        (True, False, "tilelang", True),
        (False, False, "tilelang", None),
        (False, False, "liger", None),
    ],
)
def test_output_processor_gathers_before_kernel_and_resolves_weight(
    monkeypatch,
    sequence_parallel,
    use_tied_weight,
    impl_backend,
    tensor_parallel_output_grad,
):
    hidden_states = torch.tensor([[1.0]])
    gathered_hidden_states = torch.tensor([[3.0]])
    tied_weight = torch.tensor([[5.0]])
    output_layer_weight = torch.tensor([[7.0]])
    labels = torch.tensor([0])
    events = []
    seen = {}

    def fake_gather(value, **kwargs):
        events.append("gather")
        assert value is hidden_states
        assert kwargs == {"tensor_parallel_output_grad": tensor_parallel_output_grad}
        return gathered_hidden_states

    def fake_copy(value):
        events.append("copy")
        assert value is hidden_states
        return gathered_hidden_states

    def fake_linear_cross_entropy(hidden, weight, labels_arg, temperature, reduction, group, **options):
        events.append("linear_cross_entropy")
        seen.update(
            hidden=hidden,
            weight=weight,
            labels=labels_arg,
            temperature=temperature,
            options=options,
        )
        return torch.tensor([11.0]), torch.tensor([13.0])

    monkeypatch.setattr(mff, "gather_from_sequence_parallel_region", fake_gather)
    monkeypatch.setattr(mff, "copy_to_tensor_model_parallel_region", fake_copy)
    monkeypatch.setattr(mff, "linear_cross_entropy", fake_linear_cross_entropy)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: None)

    output_weight = tied_weight if use_tied_weight else None
    output = mff.fused_output_processor(
        hidden_states=hidden_states,
        output_layer=SimpleNamespace(weight=output_layer_weight),
        output_weight=output_weight,
        labels=labels,
        context=mff.FusedOutputProcessorContext(
            temperature=0.9,
            impl_backend=impl_backend,
        ),
        config=SimpleNamespace(sequence_parallel=sequence_parallel),
    )

    uses_copy = not sequence_parallel and impl_backend != "liger"
    expected_hidden = gathered_hidden_states if sequence_parallel or uses_copy else hidden_states
    expected_weight = tied_weight if use_tied_weight else output_layer_weight
    mapping_events = ["gather"] if sequence_parallel else (["copy"] if uses_copy else [])
    assert events == mapping_events + ["linear_cross_entropy"]
    assert seen["hidden"] is expected_hidden
    assert seen["weight"] is expected_weight
    assert seen["labels"] is labels
    assert seen["temperature"] == pytest.approx(0.9)
    assert seen["options"] == {"impl_backend": impl_backend}
    assert output.log_probs.tolist() == [11.0]
    assert output.entropy.tolist() == [13.0]
