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

import importlib
from pathlib import Path

import pytest
from tensordict import NonTensorStack


@pytest.mark.parametrize(
    "module_name,test_name",
    [
        ("test_basic_agent_loop", "test_single_turn"),
        ("test_basic_agent_loop", "test_tool_agent"),
        ("test_basic_agent_loop", "test_function_tool_agent"),
        ("test_multi_modal", "test_multimodal_tool_agent"),
        ("test_multi_modal", "test_multimodal_single_turn_agent"),
    ],
)
def test_integration_inputs_have_per_prompt_batch_dimension(monkeypatch, tmp_path, module_name, test_name):
    """Exercise GPU test input construction without loading a model or starting Ray."""
    module = importlib.import_module(f"tests.experimental.agent_loop.{module_name}")
    monkeypatch.setenv("ROLLOUT_NAME", "vllm")
    monkeypatch.setattr(module.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(module.ray, "shutdown", lambda: None)
    monkeypatch.setattr(module.tq, "close", lambda: None)
    monkeypatch.setattr(module, "init_agent_loop_manager", lambda config: None)

    def open_test_file(path, mode="r", **kwargs):
        if "w" in mode:
            path = tmp_path / Path(path).name
        return open(path, mode, **kwargs)

    monkeypatch.setattr(module, "open", open_test_file, raising=False)

    class InputsChecked(Exception):
        pass

    def check_inputs(manager, batch, n):
        assert len(batch) >= 2
        for key in ["raw_prompt", "agent_name", "data_source", "reward_model"]:
            assert isinstance(batch.get(key), NonTensorStack)
            assert len(batch.get(key)) == len(batch)
        assert len(batch.get("raw_prompt")[0].data) >= 1
        raise InputsChecked

    monkeypatch.setattr(module, "generate_and_collect", check_inputs)
    config = module.init_config.__wrapped__()
    with pytest.raises(InputsChecked):
        getattr(module, test_name)(config)
