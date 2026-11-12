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
import hydra
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import _builtin_agent_loop_target, _construct_agent_loop


class _RecordingLoop:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_TARGET = f"{_RecordingLoop.__module__}.{_RecordingLoop.__qualname__}"


def test_builtin_class_path_matches_hydra_and_caches_the_class(monkeypatch):
    config = {"_target_": _TARGET}
    runtime = {"tokenizer": "tok", "marker": 3}
    via_hydra = hydra.utils.instantiate(config=config, **runtime)

    calls = []
    real_get_class = hydra.utils.get_class

    def spy(path):
        calls.append(path)
        return real_get_class(path)

    monkeypatch.setattr(hydra.utils, "get_class", spy)
    cache: dict[str, type] = {}
    first = _construct_agent_loop(config, runtime, cache)
    second = _construct_agent_loop(config, runtime, cache)

    assert _builtin_agent_loop_target(config) == _TARGET
    assert first.kwargs == second.kwargs == via_hydra.kwargs == runtime
    assert calls == [_TARGET]
    assert cache[_TARGET] is _RecordingLoop


def test_constructor_fields_stay_on_hydra():
    config = {"_target_": _TARGET, "marker": 7}
    assert _builtin_agent_loop_target(config) is None
    built = _construct_agent_loop(config, {"tokenizer": "tok"}, {})
    assert built.kwargs == {"tokenizer": "tok", "marker": 7}


def test_named_yaml_config_keeps_hydra_kwargs():
    config = OmegaConf.create({"name": "custom", "_target_": _TARGET, "marker": 1})
    assert _builtin_agent_loop_target(config) is None
    built = _construct_agent_loop(config, {"tokenizer": "tok"}, {})
    assert built.kwargs["name"] == "custom"
    assert built.kwargs["marker"] == 1
    assert built.kwargs["tokenizer"] == "tok"
