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

import math
from types import SimpleNamespace

import torch

from verl.trainer.ppo.metric_utils import (
    RolloutMoELoadBalanceMetricsAccumulator,
    compute_rollout_moe_load_balance_metrics,
    infer_moe_num_experts,
)


def test_rollout_moe_load_balance_metrics_response_tokens_only():
    routed_experts = torch.tensor(
        [
            [
                [[0, 1], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 1], [2, 2]],
                [[2, 3], [2, 2]],
                [[0, 1], [2, 2]],
                [[2, 3], [2, 2]],
            ]
        ],
        dtype=torch.long,
    )
    response_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)

    metrics = compute_rollout_moe_load_balance_metrics(
        routed_experts=routed_experts,
        response_mask=response_mask,
        num_experts=4,
    )

    assert math.isclose(metrics["rollout/moe/max_vio/layer_0"], 0.0)
    assert math.isclose(metrics["rollout/moe/min_vio/layer_0"], 0.0)
    assert math.isclose(metrics["rollout/moe/avg_vio/layer_0"], 0.0)
    assert math.isclose(metrics["rollout/moe/max_vio/layer_1"], 3.0)
    assert math.isclose(metrics["rollout/moe/min_vio/layer_1"], -1.0)
    assert math.isclose(metrics["rollout/moe/avg_vio/layer_1"], 1.5)
    assert math.isclose(metrics["rollout/moe/max_vio/max"], 3.0)
    assert math.isclose(metrics["rollout/moe/avg_vio/avg"], 0.75)


def test_rollout_moe_load_balance_metrics_skips_invalid_inputs():
    assert compute_rollout_moe_load_balance_metrics(None, torch.ones(1, 1), 4) == {}
    assert compute_rollout_moe_load_balance_metrics(torch.zeros(1, 1, 1, dtype=torch.long), torch.ones(1, 1), 4) == {}
    assert (
        compute_rollout_moe_load_balance_metrics(torch.zeros(1, 1, 1, 1, dtype=torch.long), torch.ones(1, 1), None)
        == {}
    )


def test_rollout_moe_load_balance_accumulator_spans_updates():
    accumulator = RolloutMoELoadBalanceMetricsAccumulator()
    response_mask = torch.tensor([[True]], dtype=torch.bool)

    for expert_id in range(4):
        routed_experts = torch.tensor([[[[expert_id]]]], dtype=torch.long)
        assert accumulator.update(routed_experts=routed_experts, response_mask=response_mask, num_experts=4)

    assert accumulator.total_assignments() == 4
    metrics = accumulator.pop_metrics()

    assert math.isclose(metrics["rollout/moe/max_vio/layer_0"], 0.0)
    assert math.isclose(metrics["rollout/moe/min_vio/layer_0"], 0.0)
    assert math.isclose(metrics["rollout/moe/avg_vio/layer_0"], 0.0)
    assert accumulator.total_assignments() == 0
    assert accumulator.compute() == {}


def test_infer_moe_num_experts_from_nested_config():
    assert infer_moe_num_experts({"hf_config": {"text_config": {"num_experts": 4}}}) == 4
    assert infer_moe_num_experts({"override_config": {"model_config": {"n_routed_experts": 8}}}) == 8
    assert infer_moe_num_experts({"num_local_experts": 2}) is None
    assert infer_moe_num_experts({"model_type": "mixtral", "num_local_experts": 8}) == 8
    assert infer_moe_num_experts({"model_type": "gpt_oss", "num_local_experts": 16}) == 16
    assert infer_moe_num_experts({"model_type": "qwen3_moe", "num_local_experts": 4}) is None
    assert infer_moe_num_experts({"model_type": "mixtral", "num_experts": 4, "num_local_experts": 8}) == 4


def test_trainer_infers_moe_num_experts_with_nested_override_config(monkeypatch):
    from verl.trainer.ppo.v1 import trainer_base

    class DummyTrainer(trainer_base.PPOTrainer):
        def on_step_end(self):
            return

        def on_sample_end(self):
            return

    trainer = object.__new__(DummyTrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            model={
                "path": "dummy-model",
                "override_config": {"model_config": {"max_position_embeddings": 32768}},
            }
        )
    )
    trainer._rollout_moe_num_experts_initialized = False
    trainer._rollout_moe_num_experts = None
    trainer._warned_missing_rollout_moe_lb_metrics = False

    monkeypatch.setattr(trainer_base, "copy_to_local", lambda path, use_shm=False: path)
    monkeypatch.setattr(
        trainer_base.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(num_experts=64, max_position_embeddings=8192),
    )

    assert trainer._infer_rollout_moe_num_experts() == 64
