# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import asyncio
import os

import pytest
from omegaconf import OmegaConf

from tests.experimental.agent_loop.continuous_token.compare_agentloop_ct_vs_legacy import (
    _load_tokenizer,
    _prepare_single_turn_assistant_output,
)
from tests.experimental.agent_loop.continuous_token.mock_trajectories import SINGLE_TURN_CHAT
from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.workers.rollout.replica import TokenOutput

SINGLE_TURN_MODELS_ENV = "VERL_CT_SINGLE_TURN_MODELS"


class _NoopDataset:
    pass


class _RecordingServer:
    def __init__(self, token_ids: list[int], log_probs: list[float]):
        self.token_ids = list(token_ids)
        self.log_probs = list(log_probs)
        self.prompt_ids_by_call: list[list[int]] = []

    async def generate(self, *, prompt_ids: list[int], **kwargs) -> TokenOutput:
        del kwargs
        self.prompt_ids_by_call.append(list(prompt_ids))
        return TokenOutput(
            token_ids=list(self.token_ids),
            log_probs=list(self.log_probs),
            extra_fields={"source": "mock"},
        )


def _make_config(*, model: str, enable_ct: bool):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": model, "tokenizer_path": model},
                "rollout": {
                    "prompt_length": 1024,
                    "response_length": 1024,
                    "multi_turn": {
                        "continuous_token": {
                            "enable": enable_ct,
                            "model_family": "auto",
                            "custom_builder_module": None,
                        },
                    },
                },
            }
        }
    )


def _data_config():
    return OmegaConf.create({"apply_chat_template_kwargs": {}, "mm_processor_kwargs": {}})


def _selected_models() -> list[str]:
    raw_models = os.environ.get(SINGLE_TURN_MODELS_ENV, "")
    return [model.strip() for model in raw_models.split(",") if model.strip()]


def _load_cached_model_tokenizer(model: str):
    try:
        return _load_tokenizer(model, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"Tokenizer {model!r} is not available in the local HF cache: {exc}")


async def _run_single_turn(*, model: str, tokenizer, enable_ct: bool):
    assistant_output = _prepare_single_turn_assistant_output(tokenizer, SINGLE_TURN_CHAT)
    server = _RecordingServer(assistant_output.token_ids, assistant_output.log_probs)

    loop = SingleTurnAgentLoop(
        trainer_config=DictConfigWrap(_make_config(model=model, enable_ct=enable_ct)),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=_NoopDataset,
        data_config=DictConfigWrap(_data_config()),
    )
    output = await loop.run(sampling_params={"logprobs": True}, raw_prompt=list(SINGLE_TURN_CHAT.raw_prompt))
    return loop, output, server


@pytest.mark.parametrize("model", _selected_models())
def test_single_turn_agent_loop_ct_matches_legacy(model: str):
    if not model:
        pytest.skip(
            f"Set {SINGLE_TURN_MODELS_ENV}=<model>[,<model>...] to run this standalone test, "
            "or use compare_agentloop_ct_vs_legacy.py --model <model> to run the full e2e matrix."
        )
    tokenizer = _load_cached_model_tokenizer(model)
    legacy_loop, legacy_output, legacy_server = asyncio.run(
        _run_single_turn(model=model, tokenizer=tokenizer, enable_ct=False)
    )
    ct_loop, ct_output, ct_server = asyncio.run(_run_single_turn(model=model, tokenizer=tokenizer, enable_ct=True))

    assert legacy_loop.continuous_token_builder is None
    assert ct_loop.continuous_token_builder is not None
    assert legacy_server.prompt_ids_by_call == ct_server.prompt_ids_by_call

    assert legacy_output.prompt_ids == ct_output.prompt_ids
    assert legacy_output.response_ids == ct_output.response_ids
    assert legacy_output.response_mask == ct_output.response_mask
    assert legacy_output.response_logprobs == ct_output.response_logprobs
    assert legacy_output.num_turns == ct_output.num_turns == SINGLE_TURN_CHAT.expected_num_turns
    assert legacy_output.extra_fields == ct_output.extra_fields
