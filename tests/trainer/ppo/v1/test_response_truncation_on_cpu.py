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
"""V1 must fetch truncation metadata and keep it aligned when padding rows are removed."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.trainer.ppo.v1 import trainer_base


@pytest.mark.parametrize("mtp", [False, True])
def test_v1_truncation_metadata_survives_transfer_queue_and_padding(monkeypatch, mtp):
    def jagged(lengths):
        return torch.nested.nested_tensor([torch.ones(n) for n in lengths], layout=torch.jagged)

    extras = []
    for flag in [True, True, False]:
        fields = {"response_truncated": flag}
        if mtp:
            fields.update(spec_num_draft_tokens=4, spec_num_accepted_tokens=2, spec_num_verify_steps=1)
        extras.append(NonTensorData(fields))
    data = TensorDict(
        {
            "prompts": jagged([2, 3, 4]),
            "responses": jagged([6, 5, 4]),
            **{
                key: jagged([6, 5, 4])
                for key in ("response_mask", "values", "advantages", "returns", "rm_scores", "token_level_rewards")
            },
            "num_turns": torch.tensor([2, 2, 2]),
            "extra_fields": NonTensorStack(*extras),
        },
        batch_size=[3],
    )
    selected = []

    def get_batch(*, keys, partition_id, select_fields):
        selected.append(select_fields)
        return data.select(*select_fields).clone()

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get_batch)
    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {}, "model": {"mtp": {"enable": mtp, "enable_rollout": mtp}}}}
    )
    trainer = SimpleNamespace(
        config=config, use_critic=False, _rollout_moe_lb_metrics_accumulator=None, _get_n_gpus_for_throughput=lambda: 1
    )
    batch = SimpleNamespace(
        keys=["a", "b", "padding"],
        partition_id="train",
        tags=[
            {"min_global_steps": 0, "max_global_steps": 0},
            {"min_global_steps": 0, "max_global_steps": 0},
            {"min_global_steps": 0, "max_global_steps": 0, "is_padding": True},
        ],
    )
    metrics = {}
    trainer_base.PPOTrainer._compute_metrics(trainer, batch, metrics, {"step": 1.0}, 1, 0)
    assert len(selected) == 1
    assert "extra_fields" in selected[0]
    assert metrics["response_length/clip_ratio"] == pytest.approx(1.0)
    assert metrics["response_length_non_aborted/clip_ratio"] == pytest.approx(1.0)
    if mtp:
        assert metrics["rollout/spec_accept_rate"] == pytest.approx(0.5)
