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
import asyncio
import json
import os
from typing import Any

import numpy as np
import pytest
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers.utils import get_json_schema

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer, get_trajectory_info, work_stealing_schedule
from verl.protocol import DataProto
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.tools.schemas import ToolResponse
from verl.utils import hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import CheckpointEngineConfig


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

    model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")
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


def test_tool_agent_with_interaction(init_config):
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

    interaction_config = {
        "interaction": [
            {"name": "weather", "class_name": "verl.interactions.weather_interaction.WeatherInteraction", "config": {}}
        ]
    }
    interaction_config_path = "/tmp/interaction_config.json"
    with open(interaction_config_path, "w") as f:
        json.dump(interaction_config, f)

    n = 2
    init_config.actor_rollout_ref.rollout.n = n
    init_config.actor_rollout_ref.rollout.multi_turn.tool_config_path = tool_config_path
    init_config.actor_rollout_ref.rollout.multi_turn.interaction_config_path = interaction_config_path
    init_config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 2
    agent_loop_manager = init_agent_loop_manager(init_config)
    checkpoint_engine_config = omega_conf_to_dataclass(
        init_config.actor_rollout_ref.rollout.checkpoint_engine, CheckpointEngineConfig
    )
    checkpoint_manager = CheckpointEngineManager(
        config=checkpoint_engine_config,
        trainer=agent_loop_manager.worker_group,
        replicas=agent_loop_manager.rollout_replicas,
    )
    checkpoint_manager.sleep_replicas()
    checkpoint_manager.update_weights()

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
            "extra_info": np.array(
                [
                    {"interaction_kwargs": {"name": "weather"}},
                    {"interaction_kwargs": {"name": "weather"}},
                    {"interaction_kwargs": {"name": "weather"}},
                    {"interaction_kwargs": {"name": "weather"}},
                ]
            ),
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
            # [user, assistant, user]
            assert num_turns[i] == 3
        else:
            # [user, assistant, tool, assistant, user]
            assert num_turns[i] == 5

    # Check response_mask
    tokenizer = hf_tokenizer(init_config.actor_rollout_ref.model.path)
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    attention_mask = result.batch["attention_mask"]
    assert responses.size() == response_mask.size(), f"{responses.size()} != {response_mask.size()}"
    response_length = response_mask.size(1)

    for i in range(len(responses)):
        # response with tool response
        valid_tokens = responses[i][attention_mask[i][-response_length:].bool()]
        response_with_obs = tokenizer.decode(valid_tokens)

        # response without tool response
        valid_tokens = responses[i][response_mask[i].bool()]
        response_without_obs = tokenizer.decode(valid_tokens)

        assert "\udb82\udc89" not in response_without_obs, f"found \udb82\udc89 in response: {response_without_obs}"
        assert "\udb82\udc8a" not in response_without_obs, f"found \udb82\udc8a in response: {response_without_obs}"
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
        {"step": step, "sample_index": 1, "rollout_n": 0, "validate": False},
        {"step": step, "sample_index": 1, "rollout_n": 1, "validate": False},
        {"step": step, "sample_index": 3, "rollout_n": 0, "validate": False},
        {"step": step, "sample_index": 3, "rollout_n": 1, "validate": False},
    ]

    trajectory_info = await get_trajectory_info(step, index, validate=False)

    assert trajectory_info == expected_info


# ──────────────────────────────────────────────────────────────────────
# GlobalRequestLoadBalancer unit tests (lightweight, no GPU required)
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ray_for_lb():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


class TestLoadBalancerRouting:
    """Least-loaded selection."""

    def test_distributes_across_servers(self, ray_for_lb):
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1", "s2"])
        servers = [ray.get(lb.acquire_server.remote(request_id=f"r{i}")) for i in range(3)]
        assert sorted(servers) == ["s0", "s1", "s2"]

    def test_new_requests_route_to_least_loaded(self, ray_for_lb):
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1", "s2"])
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
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        s0 = ray.get(lb.acquire_server.remote(request_id="r0"))
        s1 = ray.get(lb.acquire_server.remote(request_id="r1"))
        assert s0 != s1
        ray.get(lb.release_server.remote(server_id=s0))
        ray.get(lb.release_server.remote(server_id=s1))
        s2 = ray.get(lb.acquire_server.remote(request_id="r2"))
        s3 = ray.get(lb.acquire_server.remote(request_id="r3"))
        assert s2 != s3

    def test_release_invalid_server_raises(self, ray_for_lb):
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        with pytest.raises(ray.exceptions.RayTaskError, match="Invalid server_id") as excinfo:
            ray.get(lb.release_server.remote(server_id="nonexistent"))
        assert "Invalid server_id" in str(excinfo.value)

    def test_release_without_inflight_raises(self, ray_for_lb):
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        with pytest.raises(ray.exceptions.RayTaskError, match="no inflight") as excinfo:
            ray.get(lb.release_server.remote(server_id="s1"))
        assert "no inflight" in str(excinfo.value)


class TestLoadBalancerStickySession:
    """Request-level sticky session."""

    def test_same_request_id_same_server(self, ray_for_lb):
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1", "s2", "s3"])
        s0 = ray.get(lb.acquire_server.remote(request_id="conv-abc"))
        ray.get(lb.release_server.remote(server_id=s0))
        s1 = ray.get(lb.acquire_server.remote(request_id="conv-abc"))
        assert s0 == s1


# ──────────────────────────────────────────────────────────────────────
# Batch API unit tests (lightweight, no GPU required)
# ──────────────────────────────────────────────────────────────────────


class TestLoadBalancerBatchAcquire:
    """acquire_servers_batch / release_servers_batch — batch API for reduced RPC overhead."""

    def test_batch_distributes_across_servers(self, ray_for_lb):
        """Batch of N new requests should spread across servers just like N single calls."""
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1", "s2"])
        servers = ray.get(lb.acquire_servers_batch.remote(request_ids=["r0", "r1", "r2"]))
        assert sorted(servers) == ["s0", "s1", "s2"]

    def test_batch_respects_sticky_session(self, ray_for_lb):
        """Repeated request_ids in a batch should be sticky to the same server."""
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1", "s2"])
        # First batch: assign r0, r1 to different servers
        first = ray.get(lb.acquire_servers_batch.remote(request_ids=["r0", "r1"]))
        assert first[0] != first[1]
        # Release
        ray.get(lb.release_servers_batch.remote(server_ids=first))
        # Second batch: r0 and r1 should be sticky
        second = ray.get(lb.acquire_servers_batch.remote(request_ids=["r0", "r1"]))
        assert first == second

    def test_batch_least_loaded_within_batch(self, ray_for_lb):
        """Within a single batch call, requests should still balance across servers."""
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        # 4 new requests → should distribute 2:2 across 2 servers
        servers = ray.get(lb.acquire_servers_batch.remote(
            request_ids=["a", "b", "c", "d"]
        ))
        from collections import Counter
        counts = Counter(servers)
        assert counts["s0"] == 2
        assert counts["s1"] == 2

    def test_batch_release_multiple(self, ray_for_lb):
        """release_servers_batch should handle multiple releases including duplicates."""
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        # Acquire 3 times on s0 via sticky
        servers = ray.get(lb.acquire_servers_batch.remote(
            request_ids=["x", "x", "x"]
        ))
        assert all(s == servers[0] for s in servers)
        # Release all 3
        ray.get(lb.release_servers_batch.remote(server_ids=servers))
        # Now s0 should be at 0 inflight, so new request goes there
        s = ray.get(lb.acquire_server.remote(request_id="new"))
        # s0 had 0 inflight after release, s1 also has 0, so it could be either;
        # but the point is no error was raised during batch release
        assert s in ("s0", "s1")

    def test_batch_release_invalid_server_raises(self, ray_for_lb):
        """Batch release with an invalid server_id should raise."""
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        with pytest.raises(ray.exceptions.RayTaskError, match="Invalid server_id"):
            ray.get(lb.release_servers_batch.remote(server_ids=["nonexistent"]))

    def test_batch_empty_is_noop(self, ray_for_lb):
        """Empty batch calls should be no-ops."""
        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        result = ray.get(lb.acquire_servers_batch.remote(request_ids=[]))
        assert result == []
        # release_servers_batch with empty list should not raise
        ray.get(lb.release_servers_batch.remote(server_ids=[]))

    def test_batch_large_batch_performance(self, ray_for_lb):
        """A large batch should work correctly and be faster than individual calls."""
        import time

        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1", "s2", "s3"])
        n = 200

        # Batch version
        t0 = time.perf_counter()
        batch_result = ray.get(lb.acquire_servers_batch.remote(
            request_ids=[f"batch-{i}" for i in range(n)]
        ))
        t_batch = time.perf_counter() - t0

        assert len(batch_result) == n
        # All server_ids should be valid
        assert all(s in ("s0", "s1", "s2", "s3") for s in batch_result)

        # Clean up
        ray.get(lb.release_servers_batch.remote(server_ids=batch_result))

        # Sequential version (separate LB instance to avoid sticky interference)
        lb2 = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1", "s2", "s3"])
        t0 = time.perf_counter()
        seq_result = [
            ray.get(lb2.acquire_server.remote(request_id=f"seq-{i}"))
            for i in range(n)
        ]
        t_seq = time.perf_counter() - t0

        assert len(seq_result) == n

        # Batch should be significantly faster (at least 2x for 200 requests)
        print(f"Batch: {t_batch:.4f}s, Sequential: {t_seq:.4f}s, Speedup: {t_seq/t_batch:.1f}x")
        assert t_batch < t_seq, (
            f"Batch ({t_batch:.4f}s) should be faster than sequential ({t_seq:.4f}s)"
        )


# ──────────────────────────────────────────────────────────────────────
# Coalescer unit tests (lightweight, no GPU required)
# ──────────────────────────────────────────────────────────────────────


class TestAcquireCoalescer:
    """Tests for _AcquireCoalescer — transparent request batching in AsyncLLMServerManager."""

    @pytest.mark.asyncio
    async def test_concurrent_acquires_are_coalesced(self, ray_for_lb):
        """Multiple concurrent acquire() calls should be batched into one RPC."""
        from verl.experimental.agent_loop.agent_loop import _AcquireCoalescer

        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1", "s2", "s3"])
        coalescer = _AcquireCoalescer(lb)

        # Fire 8 concurrent acquires
        tasks = [coalescer.acquire(f"req-{i}") for i in range(8)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 8
        # All results should be valid server_ids
        assert all(s in ("s0", "s1", "s2", "s3") for s in results)
        # Should be roughly balanced: 8 requests across 4 servers → 2 each
        from collections import Counter
        counts = Counter(results)
        assert all(c == 2 for c in counts.values())

    @pytest.mark.asyncio
    async def test_coalescer_preserves_sticky_session(self, ray_for_lb):
        """Coalescer should preserve sticky session semantics."""
        from verl.experimental.agent_loop.agent_loop import _AcquireCoalescer

        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        coalescer = _AcquireCoalescer(lb)

        # First round: acquire two different requests
        results1 = await asyncio.gather(
            coalescer.acquire("conv-A"),
            coalescer.acquire("conv-B"),
        )
        assert results1[0] != results1[1]  # different conversations → different servers

        # Release both
        ray.get(lb.release_servers_batch.remote(server_ids=list(results1)))

        # Second round: same request_ids should be sticky
        results2 = await asyncio.gather(
            coalescer.acquire("conv-A"),
            coalescer.acquire("conv-B"),
        )
        assert results1 == results2

    @pytest.mark.asyncio
    async def test_coalescer_sequential_fallback(self, ray_for_lb):
        """Single acquire (no concurrency) should still work correctly."""
        from verl.experimental.agent_loop.agent_loop import _AcquireCoalescer

        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        coalescer = _AcquireCoalescer(lb)

        # Sequential calls — each should trigger its own flush
        s0 = await coalescer.acquire("solo-0")
        s1 = await coalescer.acquire("solo-1")
        assert s0 in ("s0", "s1")
        assert s1 in ("s0", "s1")


class TestReleaseCoalescer:
    """Tests for _ReleaseCoalescer."""

    @pytest.mark.asyncio
    async def test_concurrent_releases_are_batched(self, ray_for_lb):
        """Multiple release() calls should be batched into one RPC."""
        from verl.experimental.agent_loop.agent_loop import _ReleaseCoalescer

        lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        # First acquire some servers
        servers = ray.get(lb.acquire_servers_batch.remote(
            request_ids=["r0", "r1", "r2", "r3"]
        ))

        release_coalescer = _ReleaseCoalescer(lb)
        for s in servers:
            release_coalescer.release(s)

        # Wait for the fire-and-forget flush to complete
        await asyncio.sleep(0.1)

        # Verify all inflight counts are back to 0 by acquiring new requests
        # If releases didn't work, these would pile up unevenly
        new_servers = ray.get(lb.acquire_servers_batch.remote(
            request_ids=["new-0", "new-1"]
        ))
        assert sorted(new_servers) == ["s0", "s1"]


# ──────────────────────────────────────────────────────────────────────
# Work-Stealing scheduling unit tests (lightweight, no GPU required)
# ──────────────────────────────────────────────────────────────────────


@ray.remote
class MockAgentLoopWorker:
    """A mock worker that simulates generate_batch with configurable delay.

    Each call returns a DataProto-like object containing the batch size and
    an optional delay to simulate straggler behavior.
    """

    def __init__(self, worker_id: str, delay_fn=None):
        self.worker_id = worker_id
        self.delay_fn = delay_fn  # callable(batch_size) -> delay_seconds
        self.tasks_completed = 0
        self.total_samples_processed = 0

    async def generate_batch(self, batch: "DataProto") -> "DataProto":
        """Simulate processing a micro-batch."""
        batch_size = len(batch)
        if self.delay_fn:
            delay = self.delay_fn(batch_size)
        else:
            delay = 0.01 * batch_size  # 10ms per sample by default
        await asyncio.sleep(delay)
        self.tasks_completed += 1
        self.total_samples_processed += batch_size
        return batch  # Return the input as-is (passthrough for testing)

    async def generate_sequences(self, batch: "DataProto") -> "DataProto":
        """Fallback for static scheduling."""
        return await self.generate_batch(batch)

    def get_stats(self):
        return {
            "worker_id": self.worker_id,
            "tasks_completed": self.tasks_completed,
            "total_samples_processed": self.total_samples_processed,
        }


class TestWorkStealingScheduling:
    """Tests for work-stealing dynamic scheduling via the standalone work_stealing_schedule function.

    These tests call the *real* scheduling function directly, ensuring that
    any refactor or bug fix in the production code is automatically covered
    by regression tests.
    """

    def _make_fake_prompts(self, n_samples: int) -> DataProto:
        """Create a minimal DataProto with n_samples entries for testing."""
        batch = TensorDict(
            {"input_ids": torch.zeros(n_samples, 10, dtype=torch.long)},
            batch_size=n_samples,
        )
        non_tensor_batch = {
            "index": np.arange(n_samples),
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})

    @pytest.mark.asyncio
    async def test_work_stealing_processes_all_samples(self, ray_for_lb):
        """All samples should be processed exactly once."""
        n_samples = 20
        n_workers = 4
        prefetch_size = 3

        prompts = self._make_fake_prompts(n_samples)
        workers = [MockAgentLoopWorker.remote(f"w{i}") for i in range(n_workers)]

        outputs = await work_stealing_schedule(workers, prompts, prefetch_size=prefetch_size)

        # Verify: total samples across all outputs matches input
        total_output_samples = sum(len(out) for out in outputs)
        assert total_output_samples == n_samples

    @pytest.mark.asyncio
    async def test_work_stealing_preserves_sample_order(self, ray_for_lb):
        """Output samples should be in the same order as input after sorting by micro-batch id."""
        n_samples = 16
        prefetch_size = 4
        n_workers = 3

        prompts = self._make_fake_prompts(n_samples)
        workers = [MockAgentLoopWorker.remote(f"w{i}") for i in range(n_workers)]

        outputs = await work_stealing_schedule(workers, prompts, prefetch_size=prefetch_size)

        # Reconstruct the indices from ordered outputs
        reconstructed_indices = []
        for out in outputs:
            reconstructed_indices.extend(out.non_tensor_batch["index"].tolist())

        assert reconstructed_indices == list(range(n_samples))

    @pytest.mark.asyncio
    async def test_work_stealing_handles_straggler(self, ray_for_lb):
        """With a straggler worker, work-stealing should redistribute work to fast workers."""
        import time

        n_samples = 24
        prefetch_size = 4

        # Worker 0 is slow (50ms per sample), workers 1-2 are fast (5ms per sample)
        workers = [
            MockAgentLoopWorker.options().remote("slow", delay_fn=lambda bs: 0.05 * bs),
            MockAgentLoopWorker.options().remote("fast1", delay_fn=lambda bs: 0.005 * bs),
            MockAgentLoopWorker.options().remote("fast2", delay_fn=lambda bs: 0.005 * bs),
        ]

        prompts = self._make_fake_prompts(n_samples)

        t0 = time.perf_counter()
        outputs = await work_stealing_schedule(workers, prompts, prefetch_size=prefetch_size)
        t_work_stealing = time.perf_counter() - t0

        # Verify all samples processed
        total = sum(len(out) for out in outputs)
        assert total == n_samples

        # Check that fast workers did more work than the slow worker
        stats = [ray.get(w.get_stats.remote()) for w in workers]
        slow_tasks = stats[0]["total_samples_processed"]
        fast1_tasks = stats[1]["total_samples_processed"]
        fast2_tasks = stats[2]["total_samples_processed"]

        print(f"Slow worker: {slow_tasks} samples, Fast1: {fast1_tasks}, Fast2: {fast2_tasks}")
        print(f"Work-stealing time: {t_work_stealing:.3f}s")

        # Fast workers should have processed more samples than the slow worker
        assert fast1_tasks + fast2_tasks > slow_tasks, (
            f"Fast workers ({fast1_tasks}+{fast2_tasks}) should process more than slow worker ({slow_tasks})"
        )

    @pytest.mark.asyncio
    async def test_work_stealing_single_worker(self, ray_for_lb):
        """Work-stealing with a single worker should still work correctly."""
        n_samples = 8
        prefetch_size = 3

        prompts = self._make_fake_prompts(n_samples)
        workers = [MockAgentLoopWorker.remote("solo")]

        outputs = await work_stealing_schedule(workers, prompts, prefetch_size=prefetch_size)

        total = sum(len(out) for out in outputs)
        assert total == n_samples

    @pytest.mark.asyncio
    async def test_work_stealing_more_workers_than_batches(self, ray_for_lb):
        """When there are more workers than micro-batches, excess workers should stay idle."""
        n_samples = 6
        prefetch_size = 4  # → 2 micro-batches
        n_workers = 5

        prompts = self._make_fake_prompts(n_samples)
        workers = [MockAgentLoopWorker.remote(f"w{i}") for i in range(n_workers)]

        outputs = await work_stealing_schedule(workers, prompts, prefetch_size=prefetch_size)

        total = sum(len(out) for out in outputs)
        assert total == n_samples

        # Verify only 2 of 5 workers did any work
        stats = [ray.get(w.get_stats.remote()) for w in workers]
        active_workers = sum(1 for s in stats if s["tasks_completed"] > 0)
        assert active_workers == 2


class TestDataProtoSplit:
    """Tests for DataProto.split — the helper used by work-stealing scheduling."""

    def test_split_even(self):
        """split(4) on 12 samples → 3 chunks of 4."""
        batch = TensorDict(
            {"x": torch.arange(12).unsqueeze(1)},
            batch_size=12,
        )
        dp = DataProto(batch=batch, non_tensor_batch={"idx": np.arange(12)}, meta_info={})
        splits = dp.split(4)
        assert len(splits) == 3
        assert all(len(s) == 4 for s in splits)

    def test_split_uneven(self):
        """split(4) on 10 samples → 2 chunks of 4 + 1 chunk of 2."""
        batch = TensorDict(
            {"x": torch.arange(10).unsqueeze(1)},
            batch_size=10,
        )
        dp = DataProto(batch=batch, non_tensor_batch={"idx": np.arange(10)}, meta_info={})
        splits = dp.split(4)
        assert len(splits) == 3
        assert len(splits[0]) == 4
        assert len(splits[1]) == 4
        assert len(splits[2]) == 2

    def test_split_preserves_order(self):
        """Samples in splits should maintain original order."""
        batch = TensorDict(
            {"x": torch.arange(7).unsqueeze(1)},
            batch_size=7,
        )
        dp = DataProto(batch=batch, non_tensor_batch={"idx": np.arange(7)}, meta_info={})
        splits = dp.split(3)
        reconstructed = []
        for s in splits:
            reconstructed.extend(s.non_tensor_batch["idx"].tolist())
        assert reconstructed == list(range(7))
