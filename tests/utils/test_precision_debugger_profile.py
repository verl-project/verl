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

import logging
import sys
import types

from verl.utils.profiler.config import PrecisionDebuggerToolConfig
from verl.utils.profiler.precision_debugger_profile import PrecisionDebuggerProfiler


class _FakeDebugger:
    instances = []

    def __init__(self, config_path, dump_path):
        self.config_path = config_path
        self.service = types.SimpleNamespace(config=types.SimpleNamespace(dump_path=dump_path))
        self.started_models = []
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self, model):
        self.started_models.append(model)

    def stop(self):
        self.stopped = True


class _FakeModel:
    def forward(self):
        pass


def test_precision_debugger_is_created_before_first_profiled_stage(monkeypatch, tmp_path):
    """Early construction lets msprobe wrap custom APIs before model creation."""
    import verl.utils.profiler.precision_debugger_profile as profile_module

    _FakeDebugger.instances.clear()
    msprobe = types.ModuleType("msprobe")
    pytorch = types.ModuleType("msprobe.pytorch")
    pytorch.PrecisionDebugger = _FakeDebugger
    msprobe.pytorch = pytorch
    monkeypatch.setitem(sys.modules, "msprobe", msprobe)
    monkeypatch.setitem(sys.modules, "msprobe.pytorch", pytorch)
    monkeypatch.setattr(profile_module, "is_msprobe_available", lambda: True)

    profiler = PrecisionDebuggerProfiler(
        PrecisionDebuggerToolConfig(config_path="/tmp/msprobe-config.json"),
        save_path=str(tmp_path),
    )

    assert len(_FakeDebugger.instances) == 1
    debugger = _FakeDebugger.instances[0]
    assert debugger.config_path == "/tmp/msprobe-config.json"
    assert debugger.service.config.dump_path == str(tmp_path)

    model = _FakeModel()
    assert profiler.start(stage="actor_compute_log_prob", global_step=3, model=model)
    assert debugger.started_models == [model]
    assert debugger.service.config.dump_path == str(tmp_path / "step_3" / "actor_compute_log_prob")


def test_resolve_megatron_model_chunks_uses_first_valid_chunk(caplog):
    """Megatron's engine module may contain pipeline model chunks."""
    first_model = _FakeModel()
    second_model = _FakeModel()
    worker = types.SimpleNamespace(
        actor=types.SimpleNamespace(
            engine=types.SimpleNamespace(module=[object(), first_model, second_model]),
        )
    )
    profiler = PrecisionDebuggerProfiler(PrecisionDebuggerToolConfig())

    with caplog.at_level(logging.WARNING):
        model = profiler._resolve_model(worker, "actor_compute_log_prob")

    assert model is first_model
    assert "only binds the first of 2 model chunks" in caplog.text
