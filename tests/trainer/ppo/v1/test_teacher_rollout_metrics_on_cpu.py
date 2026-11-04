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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.v1 import trainer_base as module


@pytest.mark.parametrize(
    "sources,padding,fraction,staleness",
    [
        (["teacher", "student", "teacher"], [False, False, True], 0.5, 2),
        (["teacher", "teacher"], [False, False], 1.0, None),
        ([None, "student"], [False, False], 0.0, 2),
        (["teacher"], [True], None, None),
    ],
)
def test_metrics_exclude_teacher_versions_and_synthetic_rows(monkeypatch, sources, padding, fraction, staleness):
    n = len(sources)
    tags = []
    for source, is_padding in zip(sources, padding, strict=True):
        tag = {"is_padding": is_padding, "min_global_steps": 2, "max_global_steps": 2}
        if source is not None:
            tag["rollout_source"] = source
        if source == "teacher" or is_padding:
            tag.update(min_global_steps=None, max_global_steps=None)
        tags.append(tag)
    data = TensorDict(
        {
            "prompts": torch.nested.as_nested_tensor([torch.tensor([101, 102])] * n, layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor([torch.tensor([21, 22])] * n, layout=torch.jagged),
            "rm_scores": torch.nested.as_nested_tensor([torch.zeros(2)] * n, layout=torch.jagged),
            "num_turns": torch.full((n,), 2),
        },
        batch_size=[n],
    )
    monkeypatch.setattr(module, "get_metric_data_with_optional_routed_experts", lambda **kwargs: data)
    for name in (
        "compute_moe_lb_metrics",
        "compute_data_metrics",
        "compute_timing_metrics",
        "compute_throughout_metrics",
        "compute_variance_proxy_metrics",
    ):
        monkeypatch.setattr(module, name, lambda **kwargs: {})
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "distillation": {"rollout_source_selector": {"path": "my_policy.py", "name": "choose"}},
                "actor_rollout_ref": {"rollout": {}, "model": {}},
            }
        ),
        _rollout_moe_lb_metrics_accumulator=None,
        use_critic=False,
        _get_n_gpus_for_throughput=lambda: 2,
    )
    metrics = {}
    module.PPOTrainer._compute_metrics(
        trainer, SimpleNamespace(tags=tags, keys=list(range(n)), partition_id="train"), metrics, {}, 5, 0
    )
    assert metrics.get("distillation/teacher_rollout_fraction") == fraction
    assert metrics.get("training/off_policy/trajectory_staleness/mean") == staleness
    if staleness is None:
        assert not any(key.startswith("training/off_policy/") for key in metrics)
