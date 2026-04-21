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
import json
import os
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import ray
from omegaconf import DictConfig
from transformers.utils import get_json_schema

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.experimental.agent_loop.agent_loop import get_trajectory_info
from verl.experimental.agent_loop.load_balance import (
    GlobalRequestLoadBalancer,
    GroupStickyLoadBalancer,
    RequestStickyLoadBalancer,
    load_balancer_actor_class,
)
from verl.experimental.agent_loop.load_balance_strategy import (
    create_load_balance_strategy,
    host_key_for_load_balance,
)
from verl.experimental.agent_loop.prometheus_metrics import (
    collect_matching_metric_sample_lines,
    parse_prometheus_metric_value,
)
from verl.protocol import DataProto
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.tools.schemas import ToolResponse
from verl.utils import hf_tokenizer
from verl.workers.config.rollout import KvCacheMetricsConfig, RolloutConfig


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                # test sleep/wake_up with fsdp offload
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                "reward.reward_manager.name=dapo",
                "+reward.reward_kwargs.overlong_buffer_cfg.enable=False",
                "+reward.reward_kwargs.overlong_buffer_cfg.len=3072",
                "+reward.reward_kwargs.max_resp_len=4096",
            ],
        )

    model_path = os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct")
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 4
    config.actor_rollout_ref.rollout.agent.num_workers = 2
    config.actor_rollout_ref.rollout.skip_tokenizer_init = True

    return config


def test_single_turn(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    agent_loop_manager = init_agent_loop_manager(init_config)

    raw_prompts = [
        [
            {
                "role": "user",
                "content": "Let's play a role playing game. Your name is Alice, your favorite color is blue.",
            }
        ],
        [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "agent_name": np.array(["single_turn_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
        },
    )
    n = init_config.actor_rollout_ref.rollout.n
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    # check result
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len

    if init_config.actor_rollout_ref.rollout.calculate_log_probs:
        assert result.batch["rollout_log_probs"].size(1) == result.batch["responses"].size(1)

    # check compute score
    assert result.batch["rm_scores"].shape == result.batch["responses"].shape
    reward_tensor = result.batch["rm_scores"]
    reward_extra_keys = result.meta_info.get("reward_extra_keys", [])
    reward_extra_info = {key: result.non_tensor_batch[key] for key in reward_extra_keys}
    assert reward_tensor.shape == result.batch["responses"].shape
    assert "acc" in reward_extra_info, f"reward_extra_info {reward_extra_info} should contain 'acc'"
    assert reward_extra_info["acc"].shape == (len(result),), f"invalid acc: {reward_extra_info['acc']}"

    # check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    assert np.all(num_turns == 2)

    print("Test passed!")
    ray.shutdown()


class WeatherTool(BaseTool):
    def get_current_temperature(self, location: str, unit: str = "celsius"):
        """Get current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, and the unit in a dict
        """
        print(f"[DEBUG] get_current_temperature: {location}, {unit}")
        return {
            "temperature": 26.1,
            "location": location,
            "unit": unit,
        }

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_current_temperature)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            result = self.get_current_temperature(**parameters)
            return ToolResponse(text=json.dumps(result)), 0, {}
        except Exception as e:
            return ToolResponse(text=str(e)), 0, {}


class WeatherToolWithData(BaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_temperature_date)
        return OpenAIFunctionToolSchema(**schema)

    def get_temperature_date(self, location: str, date: str, unit: str = "celsius"):
        """Get temperature at a location and date.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            date: The date to get the temperature for, in the format "Year-Month-Day".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, the date and the unit in a dict
        """
        print(f"[DEBUG] get_temperature_date: {location}, {date}, {unit}")
        return {
            "temperature": 25.9,
            "location": location,
            "date": date,
            "unit": unit,
        }

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            result = self.get_temperature_date(**parameters)
            return ToolResponse(text=json.dumps(result)), 0, {}
        except Exception as e:
            return ToolResponse(text=str(e)), 0, {}


def test_tool_agent(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )

    # =========================== 1. Init rollout manager ===========================
    tool_config = {
        "tools": [
            {
                "class_name": "tests.experimental.agent_loop.test_basic_agent_loop.WeatherTool",
                "config": {"type": "native"},
            },
            {
                "class_name": "tests.experimental.agent_loop.test_basic_agent_loop.WeatherToolWithData",
                "config": {"type": "native"},
            },
        ]
    }
    tool_config_path = "/tmp/tool_config.json"
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)

    n = 2
    init_config.actor_rollout_ref.rollout.n = n
    init_config.actor_rollout_ref.rollout.multi_turn.tool_config_path = tool_config_path
    init_config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 2
    init_config.actor_rollout_ref.rollout.calculate_log_probs = True
    agent_loop_manager = init_agent_loop_manager(init_config)

    # =========================== 2. Generate sequences  ===========================
    raw_prompts = [
        [
            {"role": "user", "content": "How are you?"},
        ],
        [
            {"role": "user", "content": "What's the temperature in Los Angeles now?"},
        ],
        [
            {"role": "user", "content": "What's the temperature in New York now?"},
        ],
        [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
                "Current Date: 2024-09-30",
            },
            {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},
        ],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([np.array(prompt) for prompt in raw_prompts], dtype=object),
            "agent_name": np.array(["tool_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
        },
    )
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    # Check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    print(f"num_turns: {num_turns}")
    for i in range(len(num_turns)):
        if i // n == 0:
            # [user, assistant]
            assert num_turns[i] == 2
        else:
            # [user, assistant, tool, assistant]
            assert num_turns[i] == 4

    # Check response_mask
    tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    attention_mask = result.batch["attention_mask"]
    assert result.batch["rm_scores"].size(1) == responses.size(1)
    assert responses.size() == response_mask.size(), f"{responses.size()} != {response_mask.size()}"
    assert result.batch["rollout_log_probs"].size(1) == result.batch["responses"].size(1)

    response_length = response_mask.size(1)
    for i in range(len(responses)):
        # response with tool response
        valid_tokens = responses[i][attention_mask[i][-response_length:].bool()]
        response_with_obs = tokenizer.decode(valid_tokens)

        # response without tool response
        valid_tokens = responses[i][response_mask[i].bool()]
        response_without_obs = tokenizer.decode(valid_tokens)

        assert "<tool_response>" not in response_without_obs, (
            f"found <tool_response> in response: {response_without_obs}"
        )
        assert "</tool_response>" not in response_without_obs, (
            f"found </tool_response> in response: {response_without_obs}"
        )
        print("=========================")
        print(response_with_obs)
        print("---")
        print(response_without_obs)

    print("Test passed!")
    ray.shutdown()


@pytest.mark.asyncio
async def test_get_trajectory_info():
    """Tests the get_trajectory_info method."""
    # Initialize the class to set up class-level attributes
    step = 10
    index = [1, 1, 3, 3]
    expected_info = [
        {"step": step, "sample_index": 1, "rollout_n": 0, "validate": False, "request_group_id": "10:1"},
        {"step": step, "sample_index": 1, "rollout_n": 1, "validate": False, "request_group_id": "10:1"},
        {"step": step, "sample_index": 3, "rollout_n": 0, "validate": False, "request_group_id": "10:3"},
        {"step": step, "sample_index": 3, "rollout_n": 1, "validate": False, "request_group_id": "10:3"},
    ]

    trajectory_info = await get_trajectory_info(step, index, validate=False)

    assert trajectory_info == expected_info


# ──────────────────────────────────────────────────────────────────────
# GlobalRequestLoadBalancer unit tests (lightweight, no GPU required)
# ──────────────────────────────────────────────────────────────────────


def _rollout_for_lb(load_balance_strategy: str = "least_requests", **kwargs: Any) -> RolloutConfig:
    fields: dict[str, Any] = {"name": "vllm", "load_balance_strategy": load_balance_strategy}
    fields.update(kwargs)
    return RolloutConfig(**fields)


def _lb_remote(server_actor_ids: list[str], rollout_config: RolloutConfig):
    cls = load_balancer_actor_class(rollout_config)
    return cls.remote(server_actor_ids=server_actor_ids, rollout_config=rollout_config)


@pytest.fixture(scope="module")
def ray_for_lb():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


class TestLoadBalancerRouting:
    """Least-loaded selection."""

    def test_distributes_across_servers(self, ray_for_lb):
        lb = _lb_remote(
            server_actor_ids=["s0", "s1", "s2"],
            rollout_config=_rollout_for_lb(),
        )
        servers = [ray.get(lb.acquire_server.remote(request_id=f"r{i}")) for i in range(3)]
        assert sorted(servers) == ["s0", "s1", "s2"]

    def test_new_requests_route_to_least_loaded(self, ray_for_lb):
        lb = _lb_remote(
            server_actor_ids=["s0", "s1", "s2"],
            rollout_config=_rollout_for_lb(),
        )
        # Load s0 with 3 inflight requests
        ray.get(lb.acquire_server.remote(request_id="a"))  # -> s0
        ray.get(lb.acquire_server.remote(request_id="a"))  # sticky -> s0
        ray.get(lb.acquire_server.remote(request_id="a"))  # sticky -> s0
        # Load s1 with 1 inflight request
        ray.get(lb.acquire_server.remote(request_id="b"))  # -> s1
        # s2 has 0 inflight, so next new request must go to s2
        s_new = ray.get(lb.acquire_server.remote(request_id="d"))
        assert s_new == "s2"

    def test_release_rebalances(self, ray_for_lb):
        lb = _lb_remote(
            server_actor_ids=["s0", "s1"],
            rollout_config=_rollout_for_lb(),
        )
        s0 = ray.get(lb.acquire_server.remote(request_id="r0"))
        s1 = ray.get(lb.acquire_server.remote(request_id="r1"))
        assert s0 != s1
        ray.get(lb.release_server.remote(server_id=s0))
        ray.get(lb.release_server.remote(server_id=s1))
        s2 = ray.get(lb.acquire_server.remote(request_id="r2"))
        s3 = ray.get(lb.acquire_server.remote(request_id="r3"))
        assert s2 != s3

    def test_release_invalid_server_raises(self, ray_for_lb):
        lb = _lb_remote(
            server_actor_ids=["s0", "s1"],
            rollout_config=_rollout_for_lb(),
        )
        with pytest.raises(ray.exceptions.RayTaskError, match="Invalid server_id") as excinfo:
            ray.get(lb.release_server.remote(server_id="nonexistent"))
        assert "Invalid server_id" in str(excinfo.value)

    def test_release_without_inflight_raises(self, ray_for_lb):
        lb = _lb_remote(
            server_actor_ids=["s0", "s1"],
            rollout_config=_rollout_for_lb(),
        )
        with pytest.raises(ray.exceptions.RayTaskError, match="no inflight") as excinfo:
            ray.get(lb.release_server.remote(server_id="s1"))
        assert "no inflight" in str(excinfo.value)


class TestLoadBalancerStickySession:
    """Request-level sticky session."""

    def test_same_request_id_same_server(self, ray_for_lb):
        lb = _lb_remote(
            server_actor_ids=["s0", "s1", "s2", "s3"],
            rollout_config=_rollout_for_lb(),
        )
        s0 = ray.get(lb.acquire_server.remote(request_id="conv-abc"))
        ray.get(lb.release_server.remote(server_id=s0))
        s1 = ray.get(lb.acquire_server.remote(request_id="conv-abc"))
        assert s0 == s1


class TestPrometheusMetricsParse:
    def test_parse_by_metric_name(self):
        text = "# HELP x help\n# TYPE x gauge\nx 0.37\n"
        assert parse_prometheus_metric_value(text, metric_name="x") == pytest.approx(0.37)

    def test_multiple_series_same_name_uses_max(self):
        text = 'vllm:kv_cache_usage_perc{model_name="a"} 0.2\nvllm:kv_cache_usage_perc{model_name="b"} 0.7\n'
        assert parse_prometheus_metric_value(text, metric_name="vllm:kv_cache_usage_perc") == pytest.approx(0.7)

    def test_collect_matching_metric_sample_lines(self):
        text = 'vllm:kv_cache_usage_perc{model_name="a"} 0.2\nvllm:kv_cache_usage_perc{model_name="b"} 0.7\n'
        lines = collect_matching_metric_sample_lines(text, "vllm:kv_cache_usage_perc")
        assert lines == [
            'vllm:kv_cache_usage_perc{model_name="a"} 0.2',
            'vllm:kv_cache_usage_perc{model_name="b"} 0.7',
        ]

    def test_parse_skips_hash_lines_value_is_last_token(self):
        """Real vLLM /metrics: # HELP / # TYPE then a sample line; Ray log prefix is not in HTTP body."""
        text = (
            "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.\n"
            "# TYPE vllm:kv_cache_usage_perc gauge\n"
            'vllm:kv_cache_usage_perc{engine="0",model_name="/home/x/models/Qwen2.5-1.5B-Instruct"} '
            "0.024543294222486245\n"
        )
        assert parse_prometheus_metric_value(text, metric_name="vllm:kv_cache_usage_perc") == pytest.approx(
            0.024543294222486245
        )

    def test_parse_ignores_optional_prometheus_timestamp(self):
        """Exposition format allows ``value millis_timestamp``; value must not be the timestamp."""
        text = 'vllm:kv_cache_usage_perc{model_name="a"} 0.5 1710000000000\n'
        assert parse_prometheus_metric_value(text, metric_name="vllm:kv_cache_usage_perc") == pytest.approx(0.5)


class TestLoadBalanceRegistry:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown load_balance_strategy"):
            create_load_balance_strategy("not_a_real_strategy", server_actor_ids=["a"])


class TestLoadBalanceStrategyPickServer:
    """Verify pick_server semantics of each strategy in load_balance.py (no Ray / GPU required)."""

    def test_least_requests_prefers_lower_inflight_tie_break_sorted_id(self):
        strat = create_load_balance_strategy("least_requests", server_actor_ids=["s2", "s0", "s1"])
        strat._inflight["s0"] = 1
        strat._inflight["s1"] = 1
        strat._inflight["s2"] = 0
        assert strat.pick_server(["s0", "s1", "s2"]) == "s2"
        strat._inflight["s0"] = 0
        strat._inflight["s1"] = 0
        strat._inflight["s2"] = 0
        assert strat.pick_server(["s2", "s0", "s1"]) == "s0"

    def test_least_kv_cache_prefers_lowest_usage(self):
        strat = create_load_balance_strategy(
            "least_kv_cache",
            server_actor_ids=["s0", "s1"],
            metric_name="dummy",
        )
        strat.close()
        with patch.object(strat, "_kv_snapshot", return_value={"s0": 0.9, "s1": 0.1}):
            assert strat.pick_server(["s0", "s1"]) == "s1"

    def test_least_kv_cache_unknown_kv_falls_back_to_inflight(self):
        strat = create_load_balance_strategy("least_kv_cache", server_actor_ids=["s0", "s1"])
        strat._inflight["s0"] = 5
        strat._inflight["s1"] = 1
        assert strat.pick_server(["s0", "s1"]) == "s1"

    def test_weighted_rr_sequence_weights_1_and_3(self):
        strat = create_load_balance_strategy(
            "weighted_rr",
            server_actor_ids=["s0", "s1"],
            weights={"s0": 1.0, "s1": 3.0},
        )
        seq = [strat.pick_server(["s0", "s1"]) for _ in range(8)]
        assert seq == ["s1", "s0", "s1", "s1", "s1", "s0", "s1", "s1"]

    def test_weighted_rr_sequence_resolves_host_port_keys(self):
        strat = create_load_balance_strategy(
            "weighted_rr",
            server_actor_ids=["10.0.0.1:8000", "10.0.0.2:9000"],
            weights={"10.0.0.1": 1.0, "10.0.0.2": 3.0},
        )
        a, b = "10.0.0.1:8000", "10.0.0.2:9000"
        seq = [strat.pick_server([a, b]) for _ in range(8)]
        assert seq == [b, a, b, b, b, a, b, b]

    def test_weighted_rr_partial_weights_default_missing_host_to_one(self):
        """Omitted hosts use weight 1.0; here only s1 is boosted to match the 1:3 two-replica pattern."""
        strat = create_load_balance_strategy(
            "weighted_rr",
            server_actor_ids=["s0", "s1"],
            weights={"s1": 3.0},
        )
        seq = [strat.pick_server(["s0", "s1"]) for _ in range(8)]
        assert seq == ["s1", "s0", "s1", "s1", "s1", "s0", "s1", "s1"]

    def test_host_key_for_load_balance_ipv6_bracket(self):
        assert host_key_for_load_balance("[::1]:9090") == "::1"
        assert host_key_for_load_balance("192.168.0.1:8000") == "192.168.0.1"


class TestBuildGlobalLoadBalancerRemoteKwargs:
    def test_load_balancer_actor_class_respects_sticky_mode(self):
        rc_req = _rollout_for_lb(load_balance_sticky_mode="request")
        rc_grp = _rollout_for_lb(load_balance_sticky_mode="group")
        assert load_balancer_actor_class(rc_req) is RequestStickyLoadBalancer
        assert load_balancer_actor_class(rc_grp) is GroupStickyLoadBalancer

    def test_rollout_least_kv_cache_merges_kv_config_into_strategy_kwargs(self):
        rc = RolloutConfig(
            name="vllm",
            load_balance_strategy="least_kv_cache",
            kv_cache_metrics=KvCacheMetricsConfig(
                refresh_interval_s=1.5,
                metrics_path="/custom/metrics",
                metric_name="vllm:kv_cache_usage_perc",
                fetch_timeout_s=3.0,
            ),
        )
        kw = GlobalRequestLoadBalancer.init_bundle_from_rollout(rc)
        assert kw["load_balance_strategy"] == "least_kv_cache"
        assert kw["strategy_init_kwargs"] == {
            "metric_name": "vllm:kv_cache_usage_perc",
            "metrics_path": "/custom/metrics",
            "refresh_interval_s": 1.5,
            "fetch_timeout_s": 3.0,
        }


class TestLoadBalancerStrategies:
    def test_random_with_seed_is_deterministic(self, ray_for_lb):
        rc = _rollout_for_lb("random", load_balance_random_seed=123)
        lb = _lb_remote(
            server_actor_ids=["s0", "s1", "s2"],
            rollout_config=rc,
        )
        out = [ray.get(lb.acquire_server.remote(request_id=f"u{i}")) for i in range(6)]
        lb2 = _lb_remote(
            server_actor_ids=["s0", "s1", "s2"],
            rollout_config=rc,
        )
        out2 = [ray.get(lb2.acquire_server.remote(request_id=f"u{i}")) for i in range(6)]
        assert out == out2

    def test_weighted_rr_returns_valid_server(self, ray_for_lb):
        lb = _lb_remote(
            server_actor_ids=["s0", "s1"],
            rollout_config=_rollout_for_lb("weighted_rr", load_balance_weights={"s0": 1.0, "s1": 3.0}),
        )
        seq = [ray.get(lb.acquire_server.remote(request_id=f"w{i}")) for i in range(8)]
        assert seq == ["s1", "s0", "s1", "s1", "s1", "s0", "s1", "s1"]

    def test_least_kv_cache_routes_like_least_requests_when_metrics_disabled(self, ray_for_lb):
        """metric_name is None does not scrape HTTP, least_kv_cache falls back to least in-flight + sid."""
        lb = _lb_remote(
            server_actor_ids=["s0", "s1", "s2"],
            rollout_config=_rollout_for_lb("least_kv_cache"),
        )
        ray.get(lb.acquire_server.remote(request_id="a"))
        ray.get(lb.acquire_server.remote(request_id="a"))
        ray.get(lb.acquire_server.remote(request_id="a"))
        ray.get(lb.acquire_server.remote(request_id="b"))
        s_new = ray.get(lb.acquire_server.remote(request_id="new"))
        assert s_new == "s2"


class TestLoadBalancerGroupSticky:
    def test_same_group_same_server(self, ray_for_lb):
        lb = GroupStickyLoadBalancer.remote(
            server_actor_ids=["s0", "s1", "s2"],
            rollout_config=_rollout_for_lb("least_requests"),
        )
        g0 = "g0"
        a = ray.get(lb.acquire_server.remote(request_id="req-a", request_group_id=g0))
        b = ray.get(lb.acquire_server.remote(request_id="req-b", request_group_id=g0))
        assert a == b

    def test_different_groups_route_independently_same_request_id_ok(self, ray_for_lb):
        """Group sticky uses only request_group_id; same request_id on two groups gets two LRU entries."""
        lb = GroupStickyLoadBalancer.remote(
            server_actor_ids=["s0", "s1", "s2"],
            rollout_config=_rollout_for_lb("least_requests"),
        )
        s_g1 = ray.get(lb.acquire_server.remote(request_id="same-rid", request_group_id="g1"))
        ray.get(lb.release_server.remote(server_id=s_g1))
        s_g2 = ray.get(lb.acquire_server.remote(request_id="same-rid", request_group_id="g2"))
        ray.get(lb.release_server.remote(server_id=s_g2))
        assert ray.get(lb.acquire_server.remote(request_id="other", request_group_id="g1")) == s_g1
        assert ray.get(lb.acquire_server.remote(request_id="other", request_group_id="g2")) == s_g2
