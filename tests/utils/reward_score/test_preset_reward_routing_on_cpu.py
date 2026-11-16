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
import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.reward_score import (
    REWARD_PRESETS,
    default_compute_score,
    resolve_reward_preset,
    validate_reward_preset,
)


def test_unmapped_data_source_raises_without_preset():
    """Ensure unmapped data_source still raises NotImplementedError by default."""
    with pytest.raises(NotImplementedError, match="dapo_filter"):
        default_compute_score("dapo_filter", r"\boxed{34}", "34")


def test_fallback_preset_routes_unmapped_without_overriding_builtin_sources():
    """Verify reward.preset scores unmapped sources while preserving built-in multi-dataset validation."""
    dapo_res = default_compute_score("dapo_filter", r"\boxed{34}", "34", preset="math_dapo")
    assert dapo_res["acc"] is True and dapo_res["score"] == 1.0

    sub_source_res = default_compute_score("MATH##Aops", r"\boxed{34}", "34", preset="math_dapo")
    assert sub_source_res["acc"] is True and sub_source_res["score"] == 1.0

    # Built-in openai/gsm8k continues to use gsm8k scorer (#### format) even when fallback preset is math_dapo.
    gsm_res = default_compute_score("openai/gsm8k", "Step 1. #### 42", "42", preset="math_dapo")
    assert gsm_res == 1.0


def test_invalid_preset_or_missing_optional_dependency_fails_eagerly(monkeypatch):
    """Reject unknown preset names, non-string falsy values, and missing math-verify dependency eagerly."""
    assert "math_dapo" in REWARD_PRESETS
    for invalid in ("nonexistent_preset", "", False, 0):
        with pytest.raises(ValueError, match="Unknown preset reward function"):
            validate_reward_preset(invalid)  # type: ignore[arg-type]

    # Direct calls with a built-in data_source must still validate an invalid fallback preset upfront.
    with pytest.raises(ValueError, match="Unknown preset reward function"):
        resolve_reward_preset("openai/gsm8k", preset="nonexistent_preset")
    with pytest.raises(ValueError, match="Unknown preset reward function"):
        default_compute_score("openai/gsm8k", "#### 42", "42", preset="nonexistent_preset")

    # Selecting math_verify without optional math-verify installed must fail fast with ImportError
    # rather than silently returning 0.0 for every sample during training.
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: None if name == "math_verify" else real_find_spec(name, *args, **kwargs),
    )
    with pytest.raises(ImportError, match="pip install math-verify"):
        validate_reward_preset("math_verify")
    with pytest.raises(ImportError, match="pip install math-verify"):
        default_compute_score("dapo_filter", r"\boxed{42}", "42", preset="math_verify")

    bad_cfg = OmegaConf.create(
        {
            "reward": {
                "preset": "nonexistent_preset",
                "custom_reward_function": {"path": None, "name": "compute_score"},
                "reward_manager": {"source": "register", "name": "naive", "module": {"path": None, "name": None}},
                "sandbox_fusion": {"url": None, "max_concurrent": 64, "memory_limit_mb": 1024},
            }
        }
    )
    with pytest.raises(ValueError, match="Unknown preset reward function"):
        load_reward_manager(bad_cfg, tokenizer=None)


def test_hydra_cli_reward_preset_wires_into_load_reward_manager():
    """Verify Hydra CLI override reward.preset=math_dapo wires end-to-end through load_reward_manager."""
    config_dir = str((Path(__file__).resolve().parents[3] / "verl/trainer/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=["reward.preset=math_dapo"])

    manager = load_reward_manager(cfg, tokenizer=None)
    assert manager.compute_score("dapo_filter", r"\boxed{12}", "12")["score"] == 1.0
    assert manager.compute_score("MATH##Aops", r"\boxed{12}", "12")["score"] == 1.0
    assert manager.compute_score("openai/gsm8k", "#### 12", "12") == 1.0
