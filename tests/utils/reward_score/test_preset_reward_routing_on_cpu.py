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
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.reward_score import (
    REWARD_PRESETS,
    default_compute_score,
    normalize_data_source_presets,
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

    # Built-in openai/gsm8k continues to use gsm8k scorer (#### format) even when fallback preset is math_dapo.
    gsm_res = default_compute_score("openai/gsm8k", "Step 1. #### 42", "42", preset="math_dapo")
    assert gsm_res == 1.0


def test_data_source_presets_exact_and_glob_override_builtin_table():
    """Verify exact and fnmatch glob mappings in data_source_presets take precedence over built-in routing."""
    presets = {
        "dapo_filter": "math_dapo",
        "MATH##*": "math_dapo",
        "HuggingFaceH4/MATH-500": "math_dapo",
    }
    assert resolve_reward_preset("dapo_filter", data_source_presets=presets) == "math_dapo"
    assert resolve_reward_preset("MATH##Aops", data_source_presets=presets) == "math_dapo"
    # Built-in HuggingFaceH4/MATH-500 normally returns float 1.0 via math_reward; math_dapo returns a dict.
    overridden = default_compute_score("HuggingFaceH4/MATH-500", r"\boxed{7}", "7", data_source_presets=presets)
    assert isinstance(overridden, dict) and overridden["score"] == 1.0

    # Wildcard '*' overrides all data sources when explicitly requested.
    assert resolve_reward_preset("openai/gsm8k", data_source_presets=["*=math_reward"]) == "math_reward"


def test_invalid_preset_name_or_entry_fails_eagerly():
    """Reject unknown preset names and malformed list entries eagerly."""
    assert "math_dapo" in REWARD_PRESETS
    with pytest.raises(ValueError, match="Unknown preset reward function"):
        validate_reward_preset("nonexistent_preset")
    with pytest.raises(ValueError, match="Unknown preset reward function"):
        normalize_data_source_presets({"dapo_filter": "nonexistent_preset"})
    with pytest.raises(ValueError, match="expected 'data_source=preset'"):
        normalize_data_source_presets(["malformed_entry_without_equals"])

    bad_cfg = OmegaConf.create(
        {
            "reward": {
                "preset": "nonexistent_preset",
                "data_source_presets": {},
                "custom_reward_function": {"path": None, "name": "compute_score"},
                "reward_manager": {"source": "register", "name": "naive", "module": {"path": None, "name": None}},
                "sandbox_fusion": {"url": None, "max_concurrent": 64, "memory_limit_mb": 1024},
            }
        }
    )
    with pytest.raises(ValueError, match="Unknown preset reward function"):
        load_reward_manager(bad_cfg, tokenizer=None)


def test_hydra_cli_overrides_wire_into_load_reward_manager():
    """Verify Hydra CLI overrides for reward.preset and reward.data_source_presets (dict & list) wire end-to-end."""
    config_dir = str((Path(__file__).resolve().parents[3] / "verl/trainer/config").resolve())

    # 1. Dict-style +reward.data_source_presets.<id>=<preset>
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_dict = compose(
            config_name="ppo_trainer",
            overrides=[
                "reward.preset=math_dapo",
                "+reward.data_source_presets.custom_gsm=gsm8k",
            ],
        )
    manager_dict = load_reward_manager(cfg_dict, tokenizer=None)
    assert manager_dict.compute_score("dapo_filter", r"\boxed{12}", "12")["score"] == 1.0
    assert manager_dict.compute_score("custom_gsm", "#### 12", "12") == 1.0
    assert manager_dict.compute_score("openai/gsm8k", "#### 12", "12") == 1.0

    # 2. List-style reward.data_source_presets=["pattern=preset", ...] for keys containing '/', '#', or '*'
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_list = compose(
            config_name="ppo_trainer",
            overrides=[
                'reward.data_source_presets=["dapo_filter=math_dapo","MATH##*=math_dapo","HuggingFaceH4/MATH-500=math_dapo"]',
            ],
        )
    manager_list = load_reward_manager(cfg_list, tokenizer=None)
    assert manager_list.compute_score("dapo_filter", r"\boxed{12}", "12")["score"] == 1.0
    assert manager_list.compute_score("MATH##Aops", r"\boxed{12}", "12")["score"] == 1.0
    assert isinstance(manager_list.compute_score("HuggingFaceH4/MATH-500", r"\boxed{12}", "12"), dict)
