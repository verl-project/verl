# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""LoRA on the PPO critic (AutoModelForTokenClassification) must not use the CausalLM PEFT wrapper."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2ForTokenClassification

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _tiny_config(**kwargs):
    return Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_labels=1,
        **kwargs,
    )


def _build(module, model_type, exclude_modules=None, lora_adapter_path=None):
    model_config = SimpleNamespace(
        model_type=model_type,
        lora_adapter_path=lora_adapter_path,
        use_shm=False,
        lora_rank=4,
        lora_alpha=8,
        target_modules="all-linear",
        target_parameters=None,
        exclude_modules=exclude_modules,
    )
    engine = SimpleNamespace(model_config=model_config)
    return FSDPEngine._build_lora_module(engine, module)


def test_value_model_lora_builds_and_trains_head_fully():
    module = Qwen2ForTokenClassification(_tiny_config())
    peft_model = _build(module, "value_model")

    assert peft_model.peft_config["default"].task_type is None
    names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    # LoRA adapters on the backbone, head trained in full via modules_to_save, no LoRA on the head.
    assert any("lora_A" in n for n in names)
    assert not any("score" in n and "lora_" in n for n in names)
    assert any("score.modules_to_save" in n for n in names)

    out = peft_model(input_ids=torch.randint(0, 64, (2, 5)))
    out.logits.sum().backward()


def test_language_model_lora_keeps_causal_lm_task_type():
    peft_model = _build(Qwen2ForCausalLM(_tiny_config()), "language_model")
    assert getattr(peft_model.peft_config["default"].task_type, "value", None) == "CAUSAL_LM"


def test_value_model_lora_string_exclude_modules_stays_a_regex():
    module = Qwen2ForTokenClassification(_tiny_config())
    peft_model = _build(module, "value_model", exclude_modules=r".*\.q_proj")
    names = [n for n, _ in peft_model.named_parameters()]
    assert not any("q_proj.lora_" in n for n in names)  # user's regex still excludes q_proj
    assert any("k_proj.lora_A" in n for n in names)
    assert not any("score" in n and "lora_" in n for n in names)  # head still excluded


def test_value_model_head_survives_lora_only_save_filter_and_hf_export_names():
    from verl.utils.fsdp_utils import normalize_peft_param_name

    peft_model = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model")
    state = peft_model.state_dict()
    assert any(".modules_to_save." in k for k in state)  # what save_lora_only must keep
    exported = normalize_peft_param_name(state)
    assert "score.weight" in exported and "score.bias" in exported
    assert not any("original_module" in k or "modules_to_save" in k for k in exported)


def test_lora_only_checkpoint_keeps_trained_head_and_roundtrips():
    from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager as Mgr

    peft_model = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model")
    state = peft_model.state_dict()
    kept = {k: v for k, v in state.items() if Mgr.is_adapter_key(k)}
    assert any("score.modules_to_save.default.weight" in k for k in kept)
    assert not any("original_module" in k for k in kept)
    assert Mgr.is_lora_only_state_dict(kept)
    assert not Mgr.is_lora_only_state_dict(state)
    result = peft_model.load_state_dict(kept, strict=False)
    assert not result.unexpected_keys


def test_lora_train_meta_keeps_none_task_type():
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    peft_model = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model")
    assert FSDPCheckpointManager._get_lora_train_meta(None, peft_model)["task_type"] is None
    lm = _build(Qwen2ForCausalLM(_tiny_config()), "language_model")
    assert FSDPCheckpointManager._get_lora_train_meta(None, lm)["task_type"] == "CAUSAL_LM"


def test_merger_exports_trained_head_with_modules_to_save(tmp_path):
    import json

    from safetensors.torch import load_file

    from verl.model_merger.base_model_merger import BaseModelMerger

    peft_model = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model")
    with torch.no_grad():
        peft_model.base_model.model.score.modules_to_save["default"].weight.fill_(0.5)
    state = {k: v.clone() for k, v in peft_model.state_dict().items()}

    merger = SimpleNamespace(
        config=SimpleNamespace(target_dir=str(tmp_path), local_dir=None),
        _load_lora_train_meta=lambda: {"r": 4, "lora_alpha": 8, "task_type": None},
    )
    lora_path = BaseModelMerger.save_lora_adapter(merger, state)

    cfg = json.load(open(f"{lora_path}/adapter_config.json"))
    assert cfg["task_type"] is None and cfg["modules_to_save"] == ["score"]
    adapter = load_file(f"{lora_path}/adapter_model.safetensors")
    assert torch.allclose(
        adapter["base_model.model.score.weight"], torch.full_like(adapter["base_model.model.score.weight"], 0.5)
    )
    # exported base model state carries the trained head under its plain HF name, with no PEFT wrapper keys
    assert torch.allclose(state["score.weight"], torch.full_like(state["score.weight"], 0.5))
    assert not any("modules_to_save" in k or "original_module" in k for k in state)


def test_value_model_loads_a_causal_lm_adapter_and_trains_the_head(tmp_path):
    from peft import LoraConfig, TaskType, get_peft_model

    get_peft_model(
        Qwen2ForCausalLM(_tiny_config()),
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, target_modules="all-linear"),
    ).save_pretrained(tmp_path)

    peft_model = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model", lora_adapter_path=str(tmp_path))

    assert peft_model.peft_config["default"].task_type is None
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert any("lora_A" in n for n in trainable)
    assert any("score.modules_to_save" in n for n in trainable)
    peft_model(input_ids=torch.randint(0, 64, (2, 5))).logits.sum().backward()


def test_string_exclude_modules_with_leading_regex_flags_is_still_valid():
    peft_model = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model", exclude_modules=r"(?i).*Q_PROJ")
    names = [n for n, _ in peft_model.named_parameters()]
    assert not any("q_proj.lora_" in n for n in names)
    assert any("k_proj.lora_A" in n for n in names)
    assert not any("score" in n and "lora_" in n for n in names)


def test_verbose_regex_exclude_with_trailing_comment_is_still_valid():
    peft_model = _build(
        Qwen2ForTokenClassification(_tiny_config()), "value_model", exclude_modules=r"(?x).*\.q_proj # exclude queries"
    )
    names = [n for n, _ in peft_model.named_parameters()]
    assert not any("q_proj.lora_" in n for n in names)
    assert any("k_proj.lora_A" in n for n in names)
    assert not any("score" in n and "lora_" in n for n in names)


def test_saved_critic_adapter_reloads_without_nesting_the_head_wrapper(tmp_path):
    first = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model")
    with torch.no_grad():
        first.base_model.model.score.modules_to_save["default"].weight.fill_(0.25)
    first.save_pretrained(tmp_path)

    reloaded = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model", lora_adapter_path=str(tmp_path))

    keys = list(reloaded.state_dict())
    assert not any(
        "modules_to_save.default.modules_to_save" in k or "original_module.original_module" in k for k in keys
    )
    head = reloaded.base_model.model.score.modules_to_save["default"].weight
    assert torch.allclose(head, torch.full_like(head, 0.25))  # the saved head was restored
    assert head.requires_grad


def test_exported_critic_adapter_reloads_on_the_token_classification_model(tmp_path):
    from peft import PeftModel

    from verl.model_merger.base_model_merger import BaseModelMerger

    trained = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model")
    with torch.no_grad():
        trained.base_model.model.score.modules_to_save["default"].weight.fill_(0.5)
    state = {k: v.clone() for k, v in trained.state_dict().items()}
    merger = SimpleNamespace(
        config=SimpleNamespace(target_dir=str(tmp_path), local_dir=None),
        _load_lora_train_meta=lambda: {"r": 4, "lora_alpha": 8, "task_type": None},
    )
    lora_path = BaseModelMerger.save_lora_adapter(merger, state)

    # reloads as a plain PeftModel (no causal-LM wrapper crash) and restores the trained head
    reloaded = PeftModel.from_pretrained(Qwen2ForTokenClassification(_tiny_config()), lora_path)
    head = reloaded.base_model.model.score.modules_to_save["default"].weight
    assert torch.allclose(head, torch.full_like(head, 0.5))


@pytest.mark.parametrize(
    "pattern",
    ["(?x)\n(?i).*Q_PROJ", r"(?x) (?i).*\.q_proj", r"(?i)(?x).*\.q_proj # c", r"(?x).*\.q_proj # trailing comment"],
)
def test_exclude_regex_with_separated_or_verbose_flags_stays_valid(pattern):
    peft_model = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model", exclude_modules=pattern)
    names = [n for n, _ in peft_model.named_parameters()]
    assert not any("q_proj.lora_" in n for n in names)
    assert any("k_proj.lora_A" in n for n in names)
    assert not any("score" in n and "lora_" in n for n in names)


def test_loaded_adapter_targeting_the_head_name_does_not_put_lora_on_the_head(tmp_path):
    from peft import LoraConfig, TaskType, get_peft_model

    # source model has no `score`, so the adapter can still list it as a target
    get_peft_model(
        Qwen2ForCausalLM(_tiny_config()),
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "score"]),
    ).save_pretrained(tmp_path)

    peft_model = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model", lora_adapter_path=str(tmp_path))

    names = [n for n, _ in peft_model.named_parameters()]
    assert not any("score" in n and "lora_" in n for n in names)
    assert any("score.modules_to_save" in n for n in names)
    assert any("q_proj.lora_A" in n for n in names)


@pytest.mark.parametrize("safe_serialization", [True, False])
def test_loaded_adapter_with_head_lora_weights_is_rejected(tmp_path, safe_serialization):
    from peft import LoraConfig, get_peft_model

    # an adapter that already carries LoRA weights on a layer named like the critic head
    class _LM(Qwen2ForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            self.score = torch.nn.Linear(config.hidden_size, 1)

    get_peft_model(
        _LM(_tiny_config()), LoraConfig(r=4, lora_alpha=8, target_modules=["score", "q_proj"])
    ).save_pretrained(tmp_path, safe_serialization=safe_serialization)
    assert (tmp_path / ("adapter_model.safetensors" if safe_serialization else "adapter_model.bin")).exists()
    with pytest.raises(ValueError, match="critic head"):
        _build(Qwen2ForTokenClassification(_tiny_config()), "value_model", lora_adapter_path=str(tmp_path))
