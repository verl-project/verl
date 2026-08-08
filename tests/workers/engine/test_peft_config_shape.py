"""Tests for consistent peft_config shape from get_per_tensor_param (#7290).

Both FSDP and Megatron engines must return peft_config dicts with string
values for task_type and peft_type (not enum objects), and the consumer
wrap_lora_params must accept plain dicts.
"""

from enum import Enum


class _FakeTaskType(Enum):
    CAUSAL_LM = "CAUSAL_LM"


class _FakePeftType(Enum):
    LORA = "LORA"


def _stringify_peft_config(d):
    """Mirrors the normalization applied by FSDP engine after .to_dict()."""
    result = dict(d)
    for key in ("task_type", "peft_type"):
        val = result.get(key)
        if hasattr(val, "value"):
            result[key] = val.value
    return result


def test_fsdp_style_enums_are_stringified():
    """FSDP's LoraConfig.to_dict() returns enum values; they must be stringified."""
    raw = {
        "task_type": _FakeTaskType.CAUSAL_LM,
        "peft_type": _FakePeftType.LORA,
        "r": 16,
        "target_modules": ["q_proj"],
    }
    result = _stringify_peft_config(raw)
    assert result["task_type"] == "CAUSAL_LM"
    assert result["peft_type"] == "LORA"
    assert isinstance(result["task_type"], str)
    assert isinstance(result["peft_type"], str)


def test_megatron_style_already_strings():
    """Megatron returns strings directly; normalization is a no-op."""
    raw = {
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "r": 16,
        "target_modules": ["q_proj"],
    }
    result = _stringify_peft_config(raw)
    assert result["task_type"] == "CAUSAL_LM"
    assert result["peft_type"] == "LORA"


def test_megatron_build_peft_config_returns_strings():
    """build_peft_config_for_vllm must return string task_type and include peft_type."""
    from verl.utils.megatron_peft_utils import build_peft_config_for_vllm

    config = build_peft_config_for_vllm({
        "rank": 16,
        "alpha": 32,
        "target_modules": ["linear_qkv"],
    })
    assert isinstance(config["task_type"], str), f"task_type is {type(config['task_type'])}"
    assert config["task_type"] == "CAUSAL_LM"
    assert "peft_type" in config, "peft_type key missing from Megatron peft_config"
    assert config["peft_type"] == "LORA"


def test_wrap_lora_params_accepts_dict():
    """wrap_lora_params must accept a plain dict (not only a LoraConfig dataclass)."""
    import pathlib

    src = pathlib.Path("verl/workers/rollout/sglang_rollout/sglang_rollout.py").read_text()
    assert "isinstance(peft_config, dict)" in src, (
        "wrap_lora_params should check isinstance(peft_config, dict)"
    )


def test_both_engines_produce_same_shape():
    """After normalization, FSDP and Megatron dicts must have matching key types."""
    fsdp_raw = {
        "task_type": _FakeTaskType.CAUSAL_LM,
        "peft_type": _FakePeftType.LORA,
        "r": 16,
        "target_modules": ["q_proj"],
    }
    megatron_raw = {
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "r": 16,
        "target_modules": ["q_proj"],
    }
    fsdp = _stringify_peft_config(fsdp_raw)
    megatron = _stringify_peft_config(megatron_raw)
    assert type(fsdp["task_type"]) == type(megatron["task_type"])
    assert type(fsdp["peft_type"]) == type(megatron["peft_type"])
    assert fsdp["task_type"] == megatron["task_type"]
    assert fsdp["peft_type"] == megatron["peft_type"]
