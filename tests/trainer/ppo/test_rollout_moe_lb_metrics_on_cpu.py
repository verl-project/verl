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

import verl.trainer.ppo.metric_utils as metric_utils
from verl.trainer.ppo.metric_utils import (
    RolloutMoELoadBalanceMetricsAccumulator,
    compute_moe_lb_metrics,
    compute_rollout_moe_load_balance_metrics,
    get_hf_config_override_kwargs,
    get_metric_data_with_optional_routed_experts,
    infer_moe_num_experts,
    infer_rollout_moe_num_experts,
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


def test_rollout_moe_load_balance_accumulator_infers_num_experts(monkeypatch):
    accumulator = RolloutMoELoadBalanceMetricsAccumulator(model_config={"num_experts": 4})
    response_mask = torch.tensor([[True]], dtype=torch.bool)
    routed_experts = torch.tensor([[[[1]]]], dtype=torch.long)

    monkeypatch.setattr(
        metric_utils,
        "infer_rollout_moe_num_experts",
        lambda model_config: model_config["num_experts"],
    )

    assert accumulator.update(routed_experts=routed_experts, response_mask=response_mask)
    assert accumulator.total_assignments() == 1


def test_compute_moe_lb_metrics_accumulates_until_interval():
    accumulator = RolloutMoELoadBalanceMetricsAccumulator(model_config={"num_experts": 2})
    batch = SimpleNamespace(
        batch={
            "routed_experts": torch.tensor([[[[0]], [[1]]]], dtype=torch.long),
            "response_mask": torch.tensor([[True, True]], dtype=torch.bool),
        }
    )

    assert compute_moe_lb_metrics(batch, moe_lb_metrics_interval=2, global_steps=1, accumulator=accumulator) == {}

    metrics = compute_moe_lb_metrics(batch, moe_lb_metrics_interval=2, global_steps=2, accumulator=accumulator)

    assert metrics["rollout/moe/routed_experts_found"] == 1.0
    assert metrics["rollout/moe/routed_expert_assignments"] == 4
    assert accumulator.total_assignments() == 0


def test_get_metric_data_with_optional_routed_experts_falls_back():
    calls = []

    def fake_kv_batch_get(keys, partition_id, select_fields):
        calls.append(select_fields)
        if "routed_experts" in select_fields:
            raise ValueError("field routed_experts not found")
        return {"responses": "ok"}

    accumulator = RolloutMoELoadBalanceMetricsAccumulator()

    data = get_metric_data_with_optional_routed_experts(
        keys=["k"],
        partition_id="train",
        fields=["responses"],
        moe_lb_metrics_interval=1,
        global_steps=1,
        accumulator=accumulator,
        kv_batch_get=fake_kv_batch_get,
    )

    assert data == {"responses": "ok"}
    assert calls == [["responses", "routed_experts"], ["responses"]]
    assert accumulator.routed_experts_retry_after_step == 2
    assert "missing_routed_experts" in accumulator.warned_skip_keys

    data = get_metric_data_with_optional_routed_experts(
        keys=["k"],
        partition_id="train",
        fields=["responses"],
        moe_lb_metrics_interval=1,
        global_steps=1,
        accumulator=accumulator,
        kv_batch_get=fake_kv_batch_get,
    )

    assert data == {"responses": "ok"}
    assert calls == [["responses", "routed_experts"], ["responses"], ["responses"]]

    data = get_metric_data_with_optional_routed_experts(
        keys=["k"],
        partition_id="train",
        fields=["responses"],
        moe_lb_metrics_interval=1,
        global_steps=2,
        accumulator=accumulator,
        kv_batch_get=fake_kv_batch_get,
    )

    assert data == {"responses": "ok"}
    assert calls == [
        ["responses", "routed_experts"],
        ["responses"],
        ["responses"],
        ["responses", "routed_experts"],
        ["responses"],
    ]


def test_rollout_moe_load_balance_accumulator_uses_keyed_warnings(monkeypatch):
    accumulator = RolloutMoELoadBalanceMetricsAccumulator()
    messages = []
    monkeypatch.setattr(metric_utils.logger, "warning", lambda message: messages.append(message))

    accumulator.warn_skip_once("missing_routed_experts", "missing routed experts")
    accumulator.warn_skip_once("missing_routed_experts", "missing routed experts again")
    accumulator.warn_skip_once("num_experts_missing", "missing num experts")

    assert messages == ["missing routed experts", "missing num experts"]


def test_infer_moe_num_experts_from_nested_config():
    assert infer_moe_num_experts({"hf_config": {"text_config": {"num_experts": 4}}}) == 4
    assert infer_moe_num_experts({"override_config": {"model_config": {"n_routed_experts": 8}}}) == 8
    assert infer_moe_num_experts({"num_local_experts": 2}) is None
    assert infer_moe_num_experts({"model_type": "mixtral", "num_local_experts": 8}) == 8
    assert infer_moe_num_experts({"model_type": "gpt_oss", "num_local_experts": 16}) == 16
    assert infer_moe_num_experts({"model_type": "qwen3_moe", "num_local_experts": 4}) is None
    assert infer_moe_num_experts({"model_type": "mixtral", "num_experts": 4, "num_local_experts": 8}) == 4


def test_get_hf_config_override_kwargs_unwraps_nested_model_config():
    assert get_hf_config_override_kwargs({"model_config": {"max_position_embeddings": 32768}}) == {
        "max_position_embeddings": 32768
    }
    assert get_hf_config_override_kwargs({"max_position_embeddings": 32768}) == {"max_position_embeddings": 32768}
    assert get_hf_config_override_kwargs({}) == {}


def test_infer_rollout_moe_num_experts_with_nested_override_config(monkeypatch):
    model_config = {
        "path": "dummy-model",
        "override_config": {"model_config": {"max_position_embeddings": 32768}},
    }

    monkeypatch.setattr(metric_utils, "copy_to_local", lambda path, use_shm=False: path)

    def fake_update_model_config(config, override_config):
        config.max_position_embeddings = override_config["max_position_embeddings"]

    monkeypatch.setattr(metric_utils, "update_model_config", fake_update_model_config)
    monkeypatch.setattr(
        metric_utils.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(num_experts=64, max_position_embeddings=8192),
    )

    assert infer_rollout_moe_num_experts(model_config) == 64
