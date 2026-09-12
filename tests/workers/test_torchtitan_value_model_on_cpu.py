# Copyright 2026 Individual Contributor: Zupeng Wang
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
from safetensors.torch import save_file
from tensordict import TensorDict

pytest.importorskip("torchtitan")

from torchtitan.models.qwen3 import model_registry

from verl.utils import tensordict_utils as tu
from verl.workers.config import TorchtitanEngineConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.torchtitan import TorchTitanEngineWithLMHead, TorchTitanEngineWithValueHead
from verl.workers.engine.torchtitan.value_model import Qwen3ValueStateDictAdapter
from verl.workers.utils.padding import no_padding_2_padding


@pytest.fixture
def critic():
    engine = object.__new__(TorchTitanEngineWithValueHead)
    engine.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3", initializer_range=0.02),
        lora_rank=0,
        lora={},
        lora_adapter_path=None,
    )
    engine.engine_config = TorchtitanEngineConfig()
    return engine


def test_registration_and_scalar_head_before_model_construction(critic, monkeypatch):
    monkeypatch.setenv("VERL_ENGINE_DEVICE", "cuda")
    assert EngineRegistry.get_engine_cls("value_model", "torchtitan") is TorchTitanEngineWithValueHead
    assert EngineRegistry.get_engine_cls("language_model", "torchtitan") is TorchTitanEngineWithLMHead
    actor = model_registry("debugmodel", attn_backend="flex")
    value = critic._configure_model_spec(actor)
    assert actor.model.enable_weight_tying
    assert actor.model.lm_head.out_features == 2048
    assert not actor.model.layers[0].attention.inner_attention.kernel_options
    assert not value.model.enable_weight_tying
    assert value.model.vocab_size == actor.model.vocab_size
    with torch.device("meta"):
        model = value.model.build()
    assert model.lm_head.weight.shape == (1, 256)
    assert model.lm_head.bias.shape == (1,)
    assert model.tok_embeddings.weight.shape == (2048, 256)
    assert model.lm_head.weight is not model.tok_embeddings.weight
    assert dict(model.named_parameters())["lm_head.weight"] is model.lm_head.weight


@pytest.mark.parametrize(
    "field", ["tensor_parallel_size", "context_parallel_size", "pipeline_parallel_size", "expert_parallel_size"]
)
def test_reject_unsupported_parallelism_before_trainer(critic, field):
    critic.engine_config = TorchtitanEngineConfig(**{field: 2})
    with pytest.raises(NotImplementedError, match=field):
        critic._configure_model_spec(model_registry("debugmodel", attn_backend="flex"))


@pytest.mark.parametrize("model_type,lora_rank", [("qwen3_moe", 0), ("llama", 0), ("qwen3", 8)])
def test_reject_unsupported_model(critic, model_type, lora_rank):
    critic.model_config.hf_config.model_type = model_type
    critic.model_config.lora_rank = lora_rank
    with pytest.raises(NotImplementedError):
        critic._configure_model_spec(model_registry("debugmodel", attn_backend="flex"))


@pytest.mark.parametrize("options", [{"lora": {"rank": 8}}, {"lora_adapter_path": "/unused-adapter"}])
def test_reject_nested_lora_and_pretrained_adapter(critic, options):
    for key, value in options.items():
        setattr(critic.model_config, key, value)
    with pytest.raises(NotImplementedError, match="LoRA"):
        critic._configure_model_spec(model_registry("debugmodel", attn_backend="flex"))


def make_adapter(critic, path=None):
    spec = critic._configure_model_spec(model_registry("debugmodel", attn_backend="flex"))
    return Qwen3ValueStateDictAdapter(spec.model, str(path) if path else None)


def test_hf_roundtrip_preserves_score_and_never_uses_lm_head(critic):
    adapter = make_adapter(critic)
    embedding = torch.randn(2048, 256)
    score = torch.randn(1, 256)
    bias = torch.randn(1)
    hf = {
        "model.embed_tokens.weight": embedding,
        "score.weight": score,
        "score.bias": bias,
        "lm_head.weight": torch.randn(2048, 256),
    }
    native = adapter.from_hf(hf)
    assert native["lm_head.weight"] is score
    assert native["lm_head.bias"] is bias
    assert native["tok_embeddings.weight"] is embedding
    exported = adapter.to_hf(native)
    assert set(exported) == {"model.embed_tokens.weight", "score.weight", "score.bias"}
    assert exported["score.weight"] is score
    assert "lm_head.weight" in hf  # conversion must not mutate the caller's checkpoint


@pytest.mark.parametrize("with_score", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_initial_hf_load_and_export(critic, tmp_path, with_score, with_bias):
    import torch.distributed.checkpoint as dcp

    embedding = torch.randn(2048, 256)
    saved = {"model.embed_tokens.weight": embedding}
    if with_score:
        saved["score.weight"] = torch.randn(1, 256)
    if with_bias:
        saved["score.bias"] = torch.randn(1)
    save_file(saved, tmp_path / "model.safetensors")
    adapter = make_adapter(critic, tmp_path)
    initialized_head = torch.randn(1, 256)
    native = {
        "tok_embeddings.weight": torch.empty_like(embedding),
        "lm_head.weight": initialized_head.clone(),
        "lm_head.bias": torch.zeros(1),
    }
    with adapter.initial_hf_load(str(tmp_path)):
        requested = adapter.to_hf(native)
        dcp.load(requested, storage_reader=adapter.get_hf_storage_reader(str(tmp_path)))
        loaded = adapter.from_hf(requested)
    native.update(loaded)
    torch.testing.assert_close(native["tok_embeddings.weight"], embedding)
    torch.testing.assert_close(native["lm_head.weight"], saved.get("score.weight", initialized_head))
    torch.testing.assert_close(native["lm_head.bias"], saved.get("score.bias", torch.zeros(1)))
    assert "score.weight" in adapter.to_hf(native)


def test_missing_backbone_is_not_silently_ignored(critic, tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    save_file({"model.embed_tokens.weight": torch.ones(2048, 256)}, tmp_path / "model.safetensors")
    adapter = make_adapter(critic, tmp_path)
    with pytest.raises(CheckpointException, match="Missing key"):
        with adapter.initial_hf_load(str(tmp_path)):
            requested = adapter.to_hf({"norm.weight": torch.empty(256), "lm_head.weight": torch.empty(1, 256)})
            dcp.load(requested, storage_reader=adapter.get_hf_storage_reader(str(tmp_path)))
    assert "score.weight" in adapter.to_hf({"lm_head.weight": torch.empty(1, 256)})


def test_reject_non_scalar_hf_score(critic, tmp_path):
    save_file({"score.weight": torch.ones(2, 256)}, tmp_path / "model.safetensors")
    adapter = make_adapter(critic, tmp_path)
    with pytest.raises(ValueError, match="Expected score.weight shape"):
        with adapter.initial_hf_load(str(tmp_path)):
            pytest.fail("Invalid value head must be rejected before loading.")


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("lengths", [[5, 3], [1], [1, 3]])
def test_token_values_layout_and_gradient(critic, packed, lengths):
    ids = torch.nested.as_nested_tensor([torch.arange(n) for n in lengths], layout=torch.jagged)
    data = TensorDict({"input_ids": ids}, batch_size=[len(lengths)])
    tu.assign_non_tensor(data, use_remove_padding=packed)
    shape = (1, sum(lengths), 1) if packed else (len(lengths), max(lengths), 1)
    logits = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape).requires_grad_()
    result = critic.prepare_model_outputs(logits, {}, data)
    assert set(result) == {"values"}
    values = result["values"]
    assert torch.equal(values.offsets(), ids.offsets())
    expected = logits[0, :, 0] if packed else torch.cat([logits[i, :n, 0] for i, n in enumerate(lengths)])
    torch.testing.assert_close(values.values(), expected)
    values.values().sum().backward()
    expected_grad = torch.ones_like(logits)
    if not packed:
        for i, n in enumerate(lengths):
            expected_grad[i, n:] = 0
    torch.testing.assert_close(logits.grad, expected_grad)


def test_response_value_alignment(critic):
    ids = torch.nested.as_nested_tensor([torch.arange(5), torch.arange(3)], layout=torch.jagged)
    data = TensorDict(
        {
            "input_ids": ids,
            "prompts": torch.nested.as_nested_tensor([torch.arange(3), torch.arange(1)], layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor([torch.arange(2), torch.arange(2)], layout=torch.jagged),
        },
        batch_size=[2],
    )
    logits = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    result = critic.prepare_model_outputs(logits, {}, data)
    torch.testing.assert_close(no_padding_2_padding(result["values"], data), torch.tensor([[2.0, 3.0], [5.0, 6.0]]))
