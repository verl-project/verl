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

"""Teacher scoring inherits the student's temperature without changing sampling parameters."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.teacher_loop.teacher_manager import _get_teacher_sampling_params
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DistillationTeacherModelConfig, RolloutConfig


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5])
@pytest.mark.parametrize("use_topk", [False, True])
def test_teacher_temperature_inherits_student(temperature, use_topk, caplog):
    config_dir = Path(__file__).resolve().parents[2] / "verl/trainer/config"
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"name": "vllm", "temperature": temperature, "prompt_length": 8, "response_length": 4}
            },
            "distillation": OmegaConf.load(config_dir / "distillation/distillation.yaml"),
        }
    )
    teacher = omega_conf_to_dataclass(config.distillation.teacher_models.teacher_model)
    teacher.validate_and_prepare_for_distillation(use_topk=use_topk, topk=2)
    assert teacher.inference.temperature == temperature

    loss = SimpleNamespace(topk=2, loss_settings=SimpleNamespace(use_topk=use_topk))
    params = _get_teacher_sampling_params(teacher, loss)
    assert params["temperature"] == 1.0  # Only the unused generated token uses this parameter.
    assert params["prompt_logprobs"] == (2 if use_topk else 0)
    assert not caplog.records


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_vllm_teacher_temperature_rejected(temperature):
    teacher = DistillationTeacherModelConfig(inference=RolloutConfig(name="vllm", temperature=temperature))
    with pytest.raises(ValueError, match="teacher temperature must be finite and positive"):
        teacher.validate_and_prepare_for_distillation(use_topk=False, topk=None)


def test_other_teacher_backends_keep_temperature_warning(caplog):
    teacher = DistillationTeacherModelConfig(inference=RolloutConfig(name="sglang", temperature=0.7))
    loss = SimpleNamespace(topk=None, loss_settings=SimpleNamespace(use_topk=False))
    assert _get_teacher_sampling_params(teacher, loss)["temperature"] == 1.0
    assert "only supported for vLLM teachers" in caplog.text
