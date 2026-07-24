# Copyright 2025 Individual Contributor: albert-lv
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
"""CPU unit tests for the OpenAgora (Arena) sandbox agent loop.

The external ``openagora-sdk`` package is not required: a fake ``openagora_sdk``
module is injected into ``sys.modules`` before ``ArenaAgentLoop`` is constructed,
so the tests exercise the full ``run()`` path without an OpenAgora server.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Optional

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap, _agent_loop_registry
from verl.utils.dataset.rl_dataset import RLHFDataset


class _FakeArenaClient:
    """Stand-in for ``openagora_sdk.client.ArenaClient`` recording every call."""

    instances: list[_FakeArenaClient] = []

    def __init__(self, endpoint: str = "localhost:9090"):
        self.endpoint = endpoint
        self.create_rollout_calls: list[dict[str, Any]] = []
        self.wait_calls: list[tuple[str, float]] = []
        self.trajectory: list[dict[str, Any]] = [
            {
                "step_id": 1,
                "request": {
                    "endpoint": "/v1/chat/completions",
                    "messages_json": b'{"messages": [{"role": "user", "content": "hello"}]}',
                },
                "response": {
                    "choices_json": b'[{"message": {"role": "assistant", "content": "def add pass"}}]',
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                    "logprobs_json": b'{"content": [{"token": "def", "logprob": -0.5},'
                    b' {"token": " add", "logprob": -0.3}, {"token": " pass", "logprob": -0.2}]}',
                },
            }
        ]
        self.wait_result: dict[str, Any] = {
            "status": "success",
            "reward": 1.0,
            "verification_report": {"passed": True},
        }
        _FakeArenaClient.instances.append(self)

    def create_rollout(self, **kwargs: Any) -> dict[str, Any]:
        self.create_rollout_calls.append(kwargs)
        return {"rollout_id": "rollout-123", "proxy_url": "http://proxy:9000", "token": "tok"}

    def wait(self, rollout_id: str, poll_interval: float = 1.0, timeout: float = 3600.0) -> dict[str, Any]:
        self.wait_calls.append((rollout_id, timeout))
        return dict(self.wait_result)

    def get_trajectory(self, rollout_id: str) -> list[dict[str, Any]]:
        return self.trajectory


def _install_fake_openagora_sdk() -> None:
    """Inject a fake ``openagora_sdk`` package so tests run without the real SDK."""
    sdk_module = types.ModuleType("openagora_sdk")
    client_module = types.ModuleType("openagora_sdk.client")
    client_module.ArenaClient = _FakeArenaClient
    sdk_module.client = client_module
    sys.modules["openagora_sdk"] = sdk_module
    sys.modules["openagora_sdk.client"] = client_module


_install_fake_openagora_sdk()

from verl.experimental.agent_loop.arena_agent_loop import ArenaAgentLoop  # noqa: E402


class _FakeTokenizer:
    """Deterministic word-level tokenizer supporting the chat-template probes."""

    pad_token_id = 0
    padding_side = "right"

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        del tools, return_dict, kwargs
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        if tokenize:
            return self.encode(text)
        return text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 1
            ids.append(self._vocab[word])
        return ids


def _make_config(prompt_length: int = 64, response_length: int = 64):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                    "multi_turn": {"tool_config_path": None},
                },
                "model": {},
            },
            "data": {
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
                "continuous_token": {"enable": False, "model_family": "auto"},
            },
        }
    )


def _build_loop(monkeypatch, prompt_length: int = 64, response_length: int = 64, **env_overrides):
    env = {
        "ARENA_ENDPOINT": "localhost:9090",
        "ARENA_AGENT_IMAGE": "test-image:latest",
        "ARENA_LLM_BACKEND": "http://test:8000/v1",
        "ARENA_VERIFY_COMMAND": "true",
        "ARENA_TIMEOUT_SECONDS": "60",
    }
    env.update(env_overrides)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = _make_config(prompt_length=prompt_length, response_length=response_length)
    loop = ArenaAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=None,
        tokenizer=_FakeTokenizer(),
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
    )
    return loop, _FakeArenaClient.instances[-1]


_RAW_PROMPT = [{"role": "user", "content": "Write a function."}]


def test_arena_agent_registered_on_cpu():
    assert "arena_agent" in _agent_loop_registry
    assert _agent_loop_registry["arena_agent"]["_target_"].endswith("ArenaAgentLoop")


@pytest.mark.asyncio
async def test_run_full_pipeline_on_cpu(monkeypatch):
    # Build inside the async test so the base class captures the running event loop.
    loop, client = _build_loop(monkeypatch)
    out = await loop.run(
        sampling_params={"temperature": 0.5, "top_p": 0.9, "seed": 7},
        raw_prompt=_RAW_PROMPT,
        index=0,
        global_steps=42,
    )

    assert out.reward_score == 1.0
    assert len(out.prompt_ids) > 0
    assert len(out.prompt_ids) <= loop.prompt_length
    # Trajectory response "def add pass" tokenizes to 3 word tokens.
    assert len(out.response_ids) == 3
    assert out.response_mask == [1] * len(out.response_ids)
    assert out.response_logprobs is not None
    assert len(out.response_logprobs) == len(out.response_ids)
    assert out.num_turns == 1
    assert out.metrics.generate_sequences >= 0
    assert out.metrics.tool_calls == 1.0

    assert out.extra_fields["arena_rollout_id"] == "rollout-123"
    assert out.extra_fields["arena_status"] == "success"
    assert out.extra_fields["trajectory_steps"] == 1
    assert out.extra_fields["verification_report"] == {"passed": True}
    assert out.extra_fields["min_global_steps"] == 42
    assert out.extra_fields["max_global_steps"] == 42

    # The sandbox task embeds the rendered prompt and original messages, and
    # sampling params are forwarded to the OpenAgora proxy.
    call = client.create_rollout_calls[0]
    assert call["sampling"] == {"temperature": 0.5, "top_p": 0.9, "seed": 7}
    assert call["image"] == "test-image:latest"
    assert call["llm_backend"] == "http://test:8000/v1"
    assert call["verify"] == {"command": "true"}
    payload = json.loads(call["task_file"].decode("utf-8"))
    assert payload["messages"] == _RAW_PROMPT
    assert "Write a function." in payload["prompt"]

    # wait() was called on the created rollout id.
    assert client.wait_calls[0][0] == "rollout-123"


@pytest.mark.asyncio
async def test_run_extra_info_json_string_on_cpu(monkeypatch):
    loop, client = _build_loop(monkeypatch)
    out = await loop.run(
        sampling_params={},
        raw_prompt=_RAW_PROMPT,
        index=1,
        extra_info=json.dumps({"openagora_verify": "pytest -q"}),
    )
    assert client.create_rollout_calls[0]["verify"] == {"command": "pytest -q"}
    # No global_steps kwarg -> no policy-version tags.
    assert "min_global_steps" not in out.extra_fields
    assert "max_global_steps" not in out.extra_fields


@pytest.mark.asyncio
async def test_run_extra_info_invalid_json_string_on_cpu(monkeypatch):
    loop, client = _build_loop(monkeypatch)
    await loop.run(
        sampling_params={},
        raw_prompt=_RAW_PROMPT,
        index=2,
        extra_info="not-a-json-string",
    )
    # Invalid JSON extra_info falls back to the environment verify command.
    assert client.create_rollout_calls[0]["verify"] == {"command": "true"}


@pytest.mark.asyncio
async def test_run_extra_info_dict_and_task_file_override_on_cpu(monkeypatch):
    loop, client = _build_loop(monkeypatch)
    await loop.run(
        sampling_params={},
        raw_prompt=_RAW_PROMPT,
        index=3,
        extra_info={"task_file": "custom task content", "openagora_verify": "make check"},
    )
    call = client.create_rollout_calls[0]
    assert call["task_file"] == b"custom task content"
    assert call["verify"] == {"command": "make check"}


@pytest.mark.asyncio
async def test_verify_command_priority_on_cpu(monkeypatch):
    loop, client = _build_loop(monkeypatch, ARENA_VERIFY_COMMAND="env-cmd")

    # Per-sample openagora_verify wins over the environment fallback.
    await loop.run(
        sampling_params={},
        raw_prompt=_RAW_PROMPT,
        index=0,
        extra_info={"openagora_verify": "sample-cmd"},
    )
    assert client.create_rollout_calls[-1]["verify"] == {"command": "sample-cmd"}

    # Without the per-sample key, the environment fallback is used.
    await loop.run(sampling_params={}, raw_prompt=_RAW_PROMPT, index=1)
    assert client.create_rollout_calls[-1]["verify"] == {"command": "env-cmd"}


@pytest.mark.asyncio
async def test_run_empty_response_fallback_on_cpu(monkeypatch):
    loop, client = _build_loop(monkeypatch)
    client.trajectory = []  # agent never replied
    out = await loop.run(sampling_params={}, raw_prompt=_RAW_PROMPT, index=0)
    assert out.response_ids  # fallback text still produces tokens
    assert out.response_mask == [1] * len(out.response_ids)
    assert out.response_logprobs is None
    assert out.num_turns == 1
    assert out.extra_fields["trajectory_steps"] == 0


@pytest.mark.asyncio
async def test_run_response_truncation_on_cpu(monkeypatch):
    loop, _ = _build_loop(monkeypatch, response_length=2)
    out = await loop.run(sampling_params={}, raw_prompt=_RAW_PROMPT, index=0)
    assert len(out.response_ids) == 2
    assert out.response_mask == [1, 1]
    assert out.response_logprobs is not None
    assert len(out.response_logprobs) == 2


@pytest.mark.asyncio
async def test_run_prompt_truncation_on_cpu(monkeypatch):
    loop, _ = _build_loop(monkeypatch, prompt_length=3)
    out = await loop.run(sampling_params={}, raw_prompt=_RAW_PROMPT, index=0)
    assert len(out.prompt_ids) == 3


@pytest.mark.asyncio
async def test_run_requires_raw_prompt_on_cpu(monkeypatch):
    loop, _ = _build_loop(monkeypatch)
    with pytest.raises(ValueError, match="raw_prompt"):
        await loop.run(sampling_params={}, index=0)
