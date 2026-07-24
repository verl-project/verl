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

from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)


def _teacher():
    return DistillationTeacherModelConfig(
        key="business-data",
        model_path="/tmp/teacher",
        num_replicas=0,
    )


def test_trainer_teacher_does_not_require_a_resource_pool():
    config = DistillationConfig(
        enabled=True,
        teacher_execution="trainer",
        n_gpus_per_node=0,
        nnodes=0,
        teacher_models={"teacher_model": _teacher()},
        distillation_loss=DistillationLossConfig(loss_mode="k1", use_policy_gradient=True),
    )

    assert list(config.teacher_models) == ["default"]
    assert config.teacher_models["default"].model_path == "/tmp/teacher"


def test_trainer_teacher_rejects_topk_until_scorer_is_implemented():
    with pytest.raises(NotImplementedError, match="forward_kl_topk"):
        DistillationConfig(
            enabled=True,
            teacher_execution="trainer",
            n_gpus_per_node=0,
            nnodes=0,
            teacher_models={"teacher_model": _teacher()},
            distillation_loss=DistillationLossConfig(
                loss_mode="forward_kl_topk",
                use_policy_gradient=False,
            ),
        )
