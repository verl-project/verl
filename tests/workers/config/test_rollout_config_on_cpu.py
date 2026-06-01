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

import pytest
from omegaconf import OmegaConf

from verl.workers.config.rollout import RolloutConfig


def test_rollout_yaml_exposes_vllm_tool_calling_defaults():
    """Smoke test documented vLLM tool-calling defaults in rollout config."""
    cfg = OmegaConf.load("verl/trainer/config/rollout/rollout.yaml")

    assert cfg.engine_kwargs.vllm.enable_auto_tool_choice is False
    assert cfg.engine_kwargs.vllm.tool_call_parser is None
    assert cfg.engine_kwargs.vllm.tool_parser_plugin is None


def test_vllm_auto_tool_choice_requires_tool_call_parser():
    with pytest.raises(ValueError, match="tool_call_parser"):
        RolloutConfig(
            name="vllm",
            engine_kwargs={"vllm": {"enable_auto_tool_choice": True}},
        )


def test_vllm_auto_tool_choice_accepts_tool_call_parser():
    cfg = RolloutConfig(
        name="vllm",
        engine_kwargs={"vllm": {"enable_auto_tool_choice": True, "tool_call_parser": "hermes"}},
    )

    assert cfg.engine_kwargs["vllm"]["tool_call_parser"] == "hermes"
