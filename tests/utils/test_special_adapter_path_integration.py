# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Integration test for sglang LoRA adapter path.

Tests the two-phase weight sync (base weights then adapter deltas) that
engine_workers.update_weights() performs when lora.merge=False.

Requires 1 GPU with sglang installed.
"""

from dataclasses import asdict
from importlib.util import find_spec

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

pytestmark = pytest.mark.skipif(find_spec("sglang") is None, reason="sglang not installed")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_NAME = "verl_lora_adapter"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


@pytest.fixture(scope="module")
def peft_model(lora_config):
    """Create a peft-wrapped model (CPU, for extracting params)."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    return get_peft_model(model, lora_config)


@pytest.fixture(scope="module")
def engine():
    """Launch sglang engine with LoRA support."""
    import sglang as sgl

    eng = sgl.Engine(
        model_path=MODEL_ID,
        dtype="bfloat16",
        mem_fraction_static=0.5,
        tp_size=1,
        enable_lora=True,
        max_loras_per_batch=4,
        max_lora_rank=16,
        lora_target_modules=["q_proj", "v_proj"],
    )
    yield eng
    eng.shutdown()


def _make_prompt(tokenizer, text="What is 2+2?"):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def _extract_base_params(peft_model):
    """Extract base model params with clean HF key names.

    Mimics the base_sync_done=False path in engine_workers — sends full
    base model weights (no LoRA deltas) to the rollout engine.
    """
    params = []
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            continue
        clean = name.replace("base_model.model.", "").replace(".base_layer", "")
        params.append((clean, param.detach().clone()))
    return params


def _extract_lora_tensors(peft_model):
    """Extract LoRA adapter tensors as a list of (name, tensor) tuples.

    Mimics the base_sync_done=True path — sends adapter deltas via
    LoadLoRAAdapterFromTensorsReqInput.
    """
    from peft import get_peft_model_state_dict

    state_dict = get_peft_model_state_dict(peft_model)
    return [(k, v.detach().clone()) for k, v in state_dict.items()]


class TestBaseWeightSync:
    """Phase 1: sync base weights (base_sync_done=False path)."""

    def test_update_weights_succeeds(self, engine, peft_model):
        base_params = _extract_base_params(peft_model)
        engine.update_weights_from_tensor(named_tensors=base_params)

    def test_generation_after_base_sync(self, engine, tokenizer):
        prompt = _make_prompt(tokenizer)
        output = engine.generate(prompt, {"max_new_tokens": 32, "temperature": 0.0})
        text = output["text"]
        print(f"[base sync] Generated: {text!r}")
        assert len(text) > 0, "Empty generation after base weight sync"


class TestAdapterLoading:
    """Phase 2: load adapter deltas (base_sync_done=True path)."""

    def test_load_adapter_from_tensors(self, engine, peft_model, lora_config):
        lora_tensors = _extract_lora_tensors(peft_model)
        config_dict = {k: v for k, v in asdict(lora_config).items() if v is not None}
        engine.load_lora_adapter_from_tensors(
            lora_name=ADAPTER_NAME,
            tensors=lora_tensors,
            config_dict=config_dict,
        )

    @pytest.mark.xfail(
        reason="sglang load_lora_adapter_from_tensors doesn't populate lora_ref_cache, "
        "so _resolve_lora_path validation fails. Adapter IS loaded in TP workers. "
        "This works in verl's actual flow because verl uses the HTTP server adapter "
        "which bypasses this validation.",
        raises=Exception,
    )
    def test_generation_with_adapter(self, engine, tokenizer):
        prompt = _make_prompt(tokenizer)
        output = engine.generate(
            prompt,
            {"max_new_tokens": 32, "temperature": 0.0},
            lora_path=ADAPTER_NAME,
        )
        text = output["text"]
        print(f"[adapter gen] Generated: {text!r}")
        assert len(text) > 0, "Empty generation with adapter"


class TestAdapterLifecycle:
    """Test unload + reload cycle (simulates subsequent training iterations)."""

    def test_unload_adapter(self, engine):
        engine.unload_lora_adapter(ADAPTER_NAME)

    def test_generation_without_adapter(self, engine, tokenizer):
        """After unload, base model should still generate."""
        prompt = _make_prompt(tokenizer)
        output = engine.generate(prompt, {"max_new_tokens": 32, "temperature": 0.0})
        text = output["text"]
        print(f"[after unload] Generated: {text!r}")
        assert len(text) > 0, "Empty generation after adapter unload"

    def test_reload_adapter(self, engine, peft_model, lora_config):
        lora_tensors = _extract_lora_tensors(peft_model)
        config_dict = {k: v for k, v in asdict(lora_config).items() if v is not None}
        engine.load_lora_adapter_from_tensors(
            lora_name=ADAPTER_NAME,
            tensors=lora_tensors,
            config_dict=config_dict,
        )

    @pytest.mark.xfail(
        reason="sglang load_lora_adapter_from_tensors doesn't populate lora_ref_cache",
        raises=Exception,
    )
    def test_generation_after_reload(self, engine, tokenizer):
        prompt = _make_prompt(tokenizer)
        output = engine.generate(
            prompt,
            {"max_new_tokens": 32, "temperature": 0.0},
            lora_path=ADAPTER_NAME,
        )
        text = output["text"]
        print(f"[after reload] Generated: {text!r}")
        assert len(text) > 0, "Empty generation after adapter reload"


class TestSleepWakeCycle:
    """Test release/resume with adapter-aware tags."""

    def test_release_kv_only_keeps_weights(self, engine, tokenizer):
        """Adapter mode: release only kv_cache, keep base weights."""
        engine.release_memory_occupation(tags=["kv_cache"])
        engine.resume_memory_occupation(tags=["kv_cache"])

        # Should still generate after kv-only release/resume
        prompt = _make_prompt(tokenizer)
        output = engine.generate(prompt, {"max_new_tokens": 16, "temperature": 0.0})
        text = output["text"]
        print(f"[kv-only cycle] Generated: {text!r}")
        assert len(text) > 0, "Empty generation after kv-only release/resume"

    def test_full_release_and_resume(self, engine, tokenizer):
        """Merge/no-LoRA mode: release everything, resume everything."""
        engine.release_memory_occupation(tags=["kv_cache", "weights"])
        engine.resume_memory_occupation(tags=["kv_cache", "weights"])

        prompt = _make_prompt(tokenizer)
        output = engine.generate(prompt, {"max_new_tokens": 16, "temperature": 0.0})
        text = output["text"]
        print(f"[full cycle] Generated: {text!r}")
        assert len(text) > 0, "Empty generation after full release/resume"
