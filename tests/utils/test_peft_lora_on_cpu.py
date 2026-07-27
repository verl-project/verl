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

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen3Config

from examples.tmem.run_locomo import online_sft, online_sft_batch
from verl.utils.peft_lora import (
    copy_lora_weights,
    freeze_lora_a,
    initialize_lora_with_svd,
    iter_merged_lora_weights,
    reset_lora_b,
)


def _model(seed: int):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=16,
        intermediate_size=24,
        head_dim=8,
    )
    base = AutoModelForCausalLM.from_config(config)
    return get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=3,
            lora_alpha=6,
            target_modules=["gate_proj", "up_proj", "down_proj"],
            bias="none",
        ),
    )


def _lora_layers(model):
    return [module for module in model.modules() if hasattr(module, "lora_A") and "default" in module.lora_A]


def test_svd_initializes_effective_projection_and_freezes_a():
    model = _model(1)
    names = initialize_lora_with_svd(model, freeze_a=True)
    assert len(names) == 3

    for layer in _lora_layers(model):
        weight = layer.get_base_layer().weight.detach().float()
        expected_singular_values = torch.linalg.svdvals(weight)[:3]
        effective_a = layer.lora_A["default"].weight.detach().float() * layer.scaling["default"]
        actual_singular_values = torch.linalg.svdvals(effective_a)
        torch.testing.assert_close(actual_singular_values, expected_singular_values, rtol=2e-4, atol=2e-5)
        _, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
        expected_a = singular_values[:3, None] * vh[:3]
        torch.testing.assert_close(effective_a.mT @ effective_a, expected_a.mT @ expected_a, rtol=2e-4, atol=2e-5)
        assert not layer.lora_A["default"].weight.requires_grad
        assert layer.lora_B["default"].weight.requires_grad
        assert torch.count_nonzero(layer.lora_B["default"].weight) == 0


def test_reset_and_cross_replica_sync():
    source = _model(2)
    destination = _model(2)
    initialize_lora_with_svd(source)
    with torch.no_grad():
        for layer in _lora_layers(source):
            layer.lora_B["default"].weight.normal_()

    names = copy_lora_weights(source, destination)
    assert len(names) == 3
    for source_layer, destination_layer in zip(_lora_layers(source), _lora_layers(destination), strict=True):
        torch.testing.assert_close(source_layer.lora_A["default"].weight, destination_layer.lora_A["default"].weight)
        torch.testing.assert_close(source_layer.lora_B["default"].weight, destination_layer.lora_B["default"].weight)

    input_ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(source(input_ids).logits, destination(input_ids).logits)

    reset_lora_b(source)
    for layer in _lora_layers(source):
        assert torch.count_nonzero(layer.lora_B["default"].weight) == 0


def test_sync_between_differently_named_adapters():
    source = _model(5)
    destination = _model(5)
    destination.add_adapter("episode", destination.peft_config["default"])
    initialize_lora_with_svd(source)
    with torch.no_grad():
        for layer in _lora_layers(source):
            layer.lora_B["default"].weight.normal_()

    copy_lora_weights(source, destination, destination_adapter_name="episode")
    for source_layer, destination_layer in zip(_lora_layers(source), _lora_layers(destination), strict=True):
        torch.testing.assert_close(
            source_layer.lora_A["default"].weight,
            destination_layer.lora_A["episode"].weight,
        )
        torch.testing.assert_close(
            source_layer.lora_B["default"].weight,
            destination_layer.lora_B["episode"].weight,
        )


def test_mixed_adapter_batch_matches_individual_adapter_forwards():
    model = _model(6)
    model.add_adapter("episode", model.peft_config["default"])
    initialize_lora_with_svd(model)
    copy_lora_weights(model, model, destination_adapter_name="episode")
    with torch.no_grad():
        for layer in _lora_layers(model):
            layer.lora_B["episode"].weight.normal_()
    model.requires_grad_(False).eval()
    input_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])

    mixed_logits = model(input_ids, adapter_names=["default", "episode"]).logits
    model.set_adapter("default")
    default_logits = model(input_ids[:1]).logits
    model.set_adapter("episode")
    episode_logits = model(input_ids[1:]).logits

    torch.testing.assert_close(mixed_logits[:1], default_logits)
    torch.testing.assert_close(mixed_logits[1:], episode_logits)


def test_online_step_updates_only_active_adapter_b():
    model = _model(7)
    model.add_adapter("episode", model.peft_config["default"])
    initialize_lora_with_svd(model)
    copy_lora_weights(model, model, destination_adapter_name="episode")
    model.set_adapter("episode")
    freeze_lora_a(model, adapter_name="episode")
    before = {
        (factor, adapter, name): parameter.detach().clone()
        for name, layer in model.named_modules()
        if hasattr(layer, "lora_A")
        for factor in ("lora_A", "lora_B")
        for adapter, projection in getattr(layer, factor).items()
        for parameter in [projection.weight]
    }

    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1)
    optimizer.zero_grad()
    model(torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]])).loss.backward()
    optimizer.step()

    episode_b_changed = False
    for name, layer in model.named_modules():
        if not hasattr(layer, "lora_A"):
            continue
        for factor in ("lora_A", "lora_B"):
            for adapter, projection in getattr(layer, factor).items():
                parameter = projection.weight
                changed = not torch.equal(parameter, before[(factor, adapter, name)])
                if factor == "lora_B" and adapter == "episode":
                    episode_b_changed |= changed
                else:
                    assert not changed
    assert episode_b_changed


def test_mixed_adapter_sft_matches_independent_updates():
    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            user_token = 4 + len(messages[0]["content"]) % 8
            tokens = [1, user_token]
            if len(messages) == 2:
                tokens.extend([12 + len(messages[1]["content"]) % 8, 2])
            return torch.tensor([tokens])

    def prepare_model():
        model = _model(8)
        initialize_lora_with_svd(model)
        for adapter_name in ("first", "second"):
            model.add_adapter(adapter_name, model.peft_config["default"])
            copy_lora_weights(model, model, destination_adapter_name=adapter_name)
        return model

    args = SimpleNamespace(
        max_sft_length=32,
        learning_rate=0.05,
        epochs=2,
        batch_size=2,
        max_grad_norm=0.0,
        sft_episode_microbatch_size=2,
    )
    first_pairs = [
        {"instruction": "Who went?", "output": "Alice"},
        {"instruction": "Where?", "output": "Boston"},
        {"instruction": "When?", "output": "Yesterday"},
    ]
    second_pairs = [
        {"instruction": "What color?", "output": "Blue"},
        {"instruction": "Why?", "output": "Because"},
    ]
    independent = prepare_model()
    online_sft(independent, Tokenizer(), first_pairs, args, seed=3, adapter_name="first")
    online_sft(independent, Tokenizer(), second_pairs, args, seed=3, adapter_name="second")

    mixed = prepare_model()
    online_sft_batch(
        mixed,
        Tokenizer(),
        [("first", first_pairs), ("second", second_pairs)],
        args,
        seed=3,
    )

    for independent_layer, mixed_layer in zip(_lora_layers(independent), _lora_layers(mixed), strict=True):
        for adapter_name in ("first", "second"):
            torch.testing.assert_close(
                independent_layer.lora_B[adapter_name].weight,
                mixed_layer.lora_B[adapter_name].weight,
                rtol=2e-5,
                atol=2e-6,
            )


def test_svd_initialization_defers_meta_weights_for_fsdp_broadcast():
    model = _model(3).to(device="meta")
    names = initialize_lora_with_svd(model, allow_meta=True)
    assert len(names) == 3
    for layer in _lora_layers(model):
        assert not layer.lora_A["default"].weight.requires_grad
        assert layer.lora_B["default"].weight.requires_grad


def test_iter_merged_lora_weights_yields_only_changed_base_matrices():
    model = _model(4)
    initialize_lora_with_svd(model)
    with torch.no_grad():
        for layer in _lora_layers(model):
            layer.lora_B["default"].weight.normal_()

    merged = dict(iter_merged_lora_weights(model))
    assert len(merged) == 3
    for name, layer in (
        (name.removeprefix("base_model.model."), layer)
        for name, layer in model.named_modules()
        if hasattr(layer, "lora_A") and "default" in layer.lora_A
    ):
        expected = layer.get_base_layer().weight + layer.get_delta_weight("default")
        torch.testing.assert_close(merged[f"{name}.weight"], expected)
