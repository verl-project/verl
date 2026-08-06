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

"""Unit tests for repetition_penalty support in HFRollout.

Verifies that _generate_minibatch reads repetition_penalty from the
rollout config and passes it into GenerationConfig for all sampling modes.
"""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from transformers import GenerationConfig

from verl.workers.rollout.hf_rollout import HFRollout
from verl.workers.rollout.base import BaseRollout


def _make_config(**overrides):
    """Build an OmegaConf rollout config with sensible defaults."""
    base = {
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1.0,
        "prompt_length": 16,
        "response_length": 16,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "micro_batch_size": 128,
        "val_kwargs": {
            "top_k": -1,
            "top_p": 1.0,
            "temperature": 1.0,
            "n": 1,
            "do_sample": True,
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _make_rollout(config):
    """Create an HFRollout instance, bypassing abstract method checks
    and the parent __init__ (which requires distributed setup)."""
    with patch.multiple(HFRollout, __abstractmethods__=frozenset()), \
         patch.object(BaseRollout, "__init__", lambda self, *a, **kw: None):
        return HFRollout(module=MagicMock(), config=config)


def _capture_generation_config(rollout, meta_info):
    """Call _generate_minibatch and capture the GenerationConfig kwargs.

    We intercept GenerationConfig.__init__ to record what HFRollout passes,
    avoiding the need for a real model or GPU.
    """
    captured = {}
    original_init = GenerationConfig.__init__

    def spy_init(self, **kwargs):
        captured.update(kwargs)
        original_init(self, **kwargs)

    prompts = MagicMock()
    prompts.meta_info = meta_info
    prompts.batch = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
        "position_ids": MagicMock(),
    }
    prompts.batch["input_ids"].size.return_value = 2

    with patch.object(GenerationConfig, "__init__", spy_init):
        try:
            rollout._generate_minibatch(prompts)
        except Exception:
            pass  # Expected: no real model to call generate() on

    return captured


class TestRepetitionPenaltyInConfig:
    """Verify repetition_penalty is read from config and passed through."""

    def test_default_repetition_penalty(self):
        """Default config (1.0) should pass repetition_penalty=1.0."""
        rollout = _make_rollout(_make_config())
        captured = _capture_generation_config(
            rollout, {"do_sample": True, "validate": False, "eos_token_id": 0, "pad_token_id": 0}
        )
        assert "repetition_penalty" in captured
        assert captured["repetition_penalty"] == 1.0

    def test_custom_repetition_penalty_from_config(self):
        """Setting repetition_penalty=1.15 in config should propagate."""
        rollout = _make_rollout(_make_config(repetition_penalty=1.15))
        captured = _capture_generation_config(
            rollout, {"do_sample": True, "validate": False, "eos_token_id": 0, "pad_token_id": 0}
        )
        assert captured["repetition_penalty"] == 1.15

    def test_meta_info_overrides_config(self):
        """meta_info repetition_penalty should override the config value."""
        rollout = _make_rollout(_make_config(repetition_penalty=1.0))
        captured = _capture_generation_config(
            rollout,
            {"do_sample": True, "validate": False, "repetition_penalty": 1.3, "eos_token_id": 0, "pad_token_id": 0},
        )
        assert captured["repetition_penalty"] == 1.3

    def test_validate_mode_includes_repetition_penalty(self):
        """Validation sampling should also include repetition_penalty."""
        rollout = _make_rollout(_make_config(repetition_penalty=1.1))
        captured = _capture_generation_config(
            rollout, {"do_sample": True, "validate": True, "eos_token_id": 0, "pad_token_id": 0}
        )
        assert "repetition_penalty" in captured
        assert captured["repetition_penalty"] == 1.1

    def test_greedy_mode_omits_repetition_penalty(self):
        """Greedy decoding (do_sample=False) should not set repetition_penalty."""
        rollout = _make_rollout(_make_config(repetition_penalty=1.2))
        captured = _capture_generation_config(
            rollout, {"do_sample": False, "validate": False, "eos_token_id": 0, "pad_token_id": 0}
        )
        assert "repetition_penalty" not in captured
