# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl.single_controller import ray as ray_controller
from verl.trainer import sft_trainer_ray
from verl.utils.profiler.config import NsightToolConfig, ProfilerConfig


@pytest.fixture
def trainer(monkeypatch):
    trainer = object.__new__(sft_trainer_ray.SFTTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 1, "nnodes": 1, "device": "cuda"},
            "profiler": {"tool": "nsys", "tool_config": {"nsys": {"discrete": False}}},
            "global_profiler": {
                "global_tool_config": {
                    "nsys": {"worker_nsight_options": {"capture-range": "cudaProfilerApi", "capture-range-end": None}}
                }
            },
        }
    )
    trainer.model_config = None
    trainer.engine_config = None
    trainer.optimizer_config = None
    trainer.checkpoint_config = None
    trainer.profiler_config = ProfilerConfig(tool="nsys", tool_config={"nsys": NsightToolConfig()})
    trainer.start_profile_step = 2
    trainer.end_profile_step = 4
    monkeypatch.setattr(ray_controller, "RayResourcePool", Mock())
    monkeypatch.setattr(ray_controller, "RayClassWithInitArgs", Mock())
    trainer.worker_group_class = Mock()
    monkeypatch.setattr(ray_controller, "RayWorkerGroup", trainer.worker_group_class)
    monkeypatch.setattr(sft_trainer_ray.ray, "remote", Mock())
    return trainer


@pytest.mark.parametrize(
    "discrete,capture_end,expected",
    [
        (False, None, "repeat-shutdown:1"),
        (True, None, "repeat-shutdown:3"),
        (False, "repeat-shutdown:9", "repeat-shutdown:9"),
    ],
    ids=["continuous-interval", "one-range-per-batch", "explicit-capture-end"],
)
def test_sft_capture_count_matches_interval(trainer, discrete, capture_end, expected):
    trainer.config.profiler.tool_config.nsys.discrete = discrete
    trainer.profiler_config.tool_config["nsys"] = NsightToolConfig(discrete=discrete)
    options = trainer.config.global_profiler.global_tool_config.nsys.worker_nsight_options
    options["capture-range-end"] = capture_end

    trainer._build_engine()

    kwargs = trainer.worker_group_class.call_args.kwargs
    assert kwargs["profile_steps"] == [2, 3, 4]
    assert kwargs["worker_nsight_options"]["capture-range-end"] == expected
    assert options["capture-range-end"] == capture_end


@pytest.mark.parametrize(
    "tool,interval", [("torch", (2, 4)), ("nsys", (-1, -1))], ids=["other-tool", "disabled-interval"]
)
def test_sft_does_not_attach_nsight_without_active_nsight_interval(trainer, tool, interval):
    trainer.config.profiler.tool = tool
    trainer.start_profile_step, trainer.end_profile_step = interval

    trainer._build_engine()

    assert "worker_nsight_options" not in trainer.worker_group_class.call_args.kwargs
