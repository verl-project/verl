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
"""The FSDP engine passes packed-sequence boundaries to flash attention as explicit varlen kwargs.

Without them transformers re-derives ``cu_seqlens``/``max_seqlen`` from ``position_ids`` in every
attention layer, which costs several host-device syncs per layer. These tests check that the
engine emits the kwargs, that they describe the same segmentation as ``position_ids``, and that a
packed forward/backward with them is bitwise identical to the ``position_ids`` path, both for the
plain HF forward and for verl's fused-kernel forward.
"""

import warnings

import pytest
import torch

from verl.utils.device import get_device_name, is_device_available

if not is_device_available():
    pytest.skip("Requires an accelerator device", allow_module_level=True)

FLASH_ATTN_VARLEN_KEYS = {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}


def _make_engine(pad_to_length: bool = False, bucket: int = 16, pass_flash_attn_varlen_kwargs: bool = True):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    engine.pad_to_length = pad_to_length
    engine.pad_to_length_bucket = bucket
    engine.use_ulysses_sp = False
    if pass_flash_attn_varlen_kwargs:
        engine.pass_flash_attn_varlen_kwargs = True
    return engine


def _make_micro_batch(seqlens: list[int], device: str):
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    def nested(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    micro_batch = TensorDict(
        {
            "input_ids": nested([torch.arange(1, n + 1, device=device) for n in seqlens]),
            "position_ids": nested([torch.arange(n, device=device) for n in seqlens]),
            "temperature": torch.ones(len(seqlens), device=device),
        },
        batch_size=[len(seqlens)],
    )
    tu.assign_non_tensor(
        micro_batch, use_remove_padding=True, use_fused_kernels=False, pad_mode=DatasetPadMode.NO_PADDING
    )
    return micro_batch


def test_prepare_model_inputs_passes_flash_attn_varlen_kwargs():
    from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

    device = get_device_name()
    model_inputs, output_args = _make_engine().prepare_model_inputs(_make_micro_batch([5, 3], device))

    expected = torch.tensor([0, 5, 8], device=device, dtype=torch.int32)
    torch.testing.assert_close(model_inputs["cu_seq_lens_q"], expected)
    torch.testing.assert_close(model_inputs["cu_seq_lens_k"], expected)
    assert model_inputs["max_length_q"] == model_inputs["max_length_k"] == 5
    assert output_args["pad_size"] == 0
    # The same boundaries transformers would re-derive from position_ids in every attention layer.
    (cu_seqlens_from_position_ids, _), _ = prepare_fa_kwargs_from_position_ids(model_inputs["position_ids"])
    torch.testing.assert_close(model_inputs["cu_seq_lens_q"], cu_seqlens_from_position_ids)


def test_prepare_model_inputs_flash_attn_varlen_kwargs_cover_the_static_pad():
    from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

    device = get_device_name()
    engine = _make_engine(pad_to_length=True, bucket=16)
    model_inputs, output_args = engine.prepare_model_inputs(_make_micro_batch([5, 3], device))

    assert output_args["pad_size"] == 8
    assert model_inputs["input_ids"].shape == (1, 16)
    (cu_seqlens_from_position_ids, _), (max_length, _) = prepare_fa_kwargs_from_position_ids(
        model_inputs["position_ids"]
    )
    torch.testing.assert_close(model_inputs["cu_seq_lens_q"], cu_seqlens_from_position_ids)
    torch.testing.assert_close(
        model_inputs["cu_seq_lens_q"], torch.tensor([0, 5, 8, 16], device=device, dtype=torch.int32)
    )
    assert model_inputs["max_length_q"] == int(max_length) == 8


def test_prepare_model_inputs_skips_flash_attn_varlen_kwargs_when_unsupported():
    engine = _make_engine(pass_flash_attn_varlen_kwargs=False)
    model_inputs, _ = engine.prepare_model_inputs(_make_micro_batch([5, 3], get_device_name()))

    assert not FLASH_ATTN_VARLEN_KEYS & model_inputs.keys()


def _tiny_qwen2(device: str, use_fused_kernels: bool = False):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from verl.models.transformers.monkey_patch import apply_monkey_patch

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attn_implementation="flash_attention_2",
    )
    model = Qwen2ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    apply_monkey_patch(
        model=model,
        use_remove_padding=True,
        use_fused_kernels=use_fused_kernels,
        fused_kernels_backend="torch" if use_fused_kernels else None,
    )
    return model


@pytest.mark.parametrize("seqlens", [[37, 5, 64, 1, 20], [48]])
@pytest.mark.parametrize("use_fused_kernels", [False, True], ids=["hf_forward", "fused_forward"])
def test_flash_attn_varlen_kwargs_match_position_ids_path(seqlens, use_fused_kernels, monkeypatch):
    """A packed forward/backward with the explicit kwargs is bitwise identical to the one where the
    boundaries are re-derived from position_ids in every layer."""
    pytest.importorskip("flash_attn")
    from verl.workers.engine.utils import build_flash_attn_varlen_kwargs

    # flash-attn's backward accumulates dQ with atomics unless it runs in deterministic mode.
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    device = get_device_name()
    model = _tiny_qwen2(device, use_fused_kernels=use_fused_kernels)
    vocab_size = model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (1, sum(seqlens)), device=device)
    position_ids = torch.cat([torch.arange(n, device=device) for n in seqlens]).unsqueeze(0)
    cu_seqlens = torch.tensor([0, *seqlens], device=device).cumsum(0)

    def run(**kwargs):
        model.zero_grad(set_to_none=True)
        if use_fused_kernels:
            out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False, return_dict=True, **kwargs)
            result = out.log_probs
        else:
            result = model(input_ids=input_ids, position_ids=position_ids, use_cache=False, **kwargs).logits
        result.float().square().mean().backward()
        return result.detach(), {name: p.grad.clone() for name, p in model.named_parameters()}

    ref_out, ref_grads = run()
    out, grads = run(**build_flash_attn_varlen_kwargs(cu_seqlens))

    assert torch.equal(out, ref_out)
    for name, grad in grads.items():
        assert torch.equal(grad, ref_grads[name]), name

    # The kwargs do reach the attention layers: merging all sequences into one changes the result.
    merged = build_flash_attn_varlen_kwargs(torch.tensor([0, sum(seqlens)], device=device))
    if len(seqlens) > 1:
        assert not torch.equal(run(**merged)[0], ref_out)
