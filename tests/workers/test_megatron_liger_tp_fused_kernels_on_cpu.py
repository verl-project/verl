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

from verl.models.mcore import model_forward_fused
from verl.utils.kernel import linear_cross_entropy
from verl.workers.engine.megatron import transformer_impl


@pytest.mark.parametrize(
    ("use_liger", "expected_backend"),
    [
        (False, "triton"),
        (True, "liger_tp"),
    ],
)
def test_resolve_megatron_fused_kernel_backend(use_liger, expected_backend):
    model_config = SimpleNamespace(use_liger=use_liger)

    assert transformer_impl._resolve_megatron_fused_kernel_backend(model_config) == expected_backend


def test_configure_liger_tp_runtime_uses_existing_engine_limits(monkeypatch):
    weight = torch.empty(62080, 5120)
    model = SimpleNamespace(
        post_process=True,
        output_layer=SimpleNamespace(weight=weight),
    )
    engine_config = SimpleNamespace(
        max_token_len_per_gpu=4096,
        infer_max_token_len_per_gpu=2048,
        context_parallel_size=2,
    )
    process_group = object()
    calls = []

    monkeypatch.setattr(transformer_impl, "unwrap_model", lambda value: value)
    monkeypatch.setattr(transformer_impl.mpu, "get_tensor_model_parallel_group", lambda: process_group)
    monkeypatch.setattr(
        linear_cross_entropy,
        "configure_liger_tp_flsce",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    assert transformer_impl._configure_liger_tp_runtime(model, engine_config) is True
    assert calls == [
        {
            "max_tokens": 8192,
            "hidden_size": 5120,
            "local_vocab_size": 62080,
            "process_group": process_group,
            "device": weight.device,
        }
    ]


def test_configure_liger_tp_runtime_resolves_tied_output_weight(monkeypatch):
    weight = torch.empty(151936, 2048)
    model = SimpleNamespace(
        post_process=True,
        output_layer=SimpleNamespace(weight=None),
        share_embeddings_and_output_weights=True,
        shared_embedding_or_output_weight=lambda: weight,
    )
    engine_config = SimpleNamespace(
        max_token_len_per_gpu=4096,
        infer_max_token_len_per_gpu=None,
        context_parallel_size=1,
    )
    process_group = object()
    calls = []

    monkeypatch.setattr(transformer_impl, "unwrap_model", lambda value: value)
    monkeypatch.setattr(transformer_impl.mpu, "get_tensor_model_parallel_group", lambda: process_group)
    monkeypatch.setattr(
        linear_cross_entropy,
        "configure_liger_tp_flsce",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    assert transformer_impl._configure_liger_tp_runtime(model, engine_config) is True
    assert calls[0]["hidden_size"] == 2048
    assert calls[0]["local_vocab_size"] == 151936
    assert calls[0]["device"] == weight.device


def test_megatron_engine_patches_liger_tp(monkeypatch):
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = SimpleNamespace(
        use_fused_kernels=True,
        use_remove_padding=True,
        max_token_len_per_gpu=4096,
        infer_max_token_len_per_gpu=2048,
        context_parallel_size=1,
    )
    engine.model_config = SimpleNamespace(
        use_liger=True,
        mtp=SimpleNamespace(enable=False),
    )
    engine.is_value_model = False
    engine.param_dtype = torch.bfloat16
    engine.module = [
        SimpleNamespace(post_process=True),
        SimpleNamespace(post_process=True),
    ]
    patch_calls = []
    configure_calls = []

    monkeypatch.setattr(
        model_forward_fused,
        "patch_fused_forward",
        lambda model, **kwargs: patch_calls.append((model, kwargs)),
    )
    monkeypatch.setattr(
        transformer_impl,
        "_configure_liger_tp_runtime",
        lambda model, engine_config: configure_calls.append((model, engine_config)) or True,
    )

    engine._maybe_enable_fused_kernels()

    assert configure_calls == [(engine.module[0], engine.engine_config)]
    assert patch_calls == [
        (
            engine.module[0],
            {"impl_backend": "liger_tp"},
        ),
        (
            engine.module[1],
            {"impl_backend": "liger_tp"},
        ),
    ]


def test_megatron_engine_patches_liger_tp_without_output_head(monkeypatch):
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = SimpleNamespace(
        use_fused_kernels=True,
        use_remove_padding=True,
        max_token_len_per_gpu=4096,
        infer_max_token_len_per_gpu=2048,
        context_parallel_size=1,
    )
    engine.model_config = SimpleNamespace(
        use_liger=True,
        mtp=SimpleNamespace(enable=False),
    )
    engine.is_value_model = False
    engine.param_dtype = torch.bfloat16
    engine.module = [SimpleNamespace(post_process=False)]
    patch_calls = []
    configure_calls = []

    monkeypatch.setattr(
        model_forward_fused,
        "patch_fused_forward",
        lambda model, **kwargs: patch_calls.append((model, kwargs)),
    )
    monkeypatch.setattr(
        transformer_impl,
        "_configure_liger_tp_runtime",
        lambda model, engine_config: configure_calls.append((model, engine_config)) or False,
    )

    engine._maybe_enable_fused_kernels()

    assert configure_calls == [(engine.module[0], engine.engine_config)]
    assert patch_calls == [
        (
            engine.module[0],
            {"impl_backend": "liger_tp"},
        )
    ]


def test_megatron_engine_allows_public_liger_fallback_for_non_bf16(monkeypatch):
    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = SimpleNamespace(
        use_fused_kernels=True,
        use_remove_padding=True,
        max_token_len_per_gpu=4096,
        infer_max_token_len_per_gpu=2048,
        context_parallel_size=1,
    )
    engine.model_config = SimpleNamespace(
        use_liger=True,
        mtp=SimpleNamespace(enable=False),
    )
    engine.is_value_model = False
    engine.param_dtype = torch.float16
    engine.module = [SimpleNamespace(post_process=False)]
    patch_calls = []
    configure_calls = []

    monkeypatch.setattr(
        model_forward_fused,
        "patch_fused_forward",
        lambda model, **kwargs: patch_calls.append((model, kwargs)),
    )
    monkeypatch.setattr(
        transformer_impl,
        "_configure_liger_tp_runtime",
        lambda model, engine_config: configure_calls.append((model, engine_config)) or True,
    )

    engine._maybe_enable_fused_kernels()

    assert configure_calls == []
    assert patch_calls == [
        (
            engine.module[0],
            {"impl_backend": "liger_tp"},
        )
    ]
