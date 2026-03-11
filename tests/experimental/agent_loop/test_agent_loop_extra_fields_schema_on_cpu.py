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

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopWorker,
    DictConfigWrap,
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeServerManager:
    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del request_id, sampling_params, image_data, video_data
        # Return a short, deterministic "generation" for testing.
        return TokenOutput(token_ids=prompt_ids[-1:] + [11, 12, 13], log_probs=[0.0, 0.0, 0.0, 0.0])

    async def generate_for_partial(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> tuple[list[int], list[float], bool]:
        del request_id, sampling_params, image_data, video_data
        # Return a short partial generation and "not cancelled".
        response_ids = prompt_ids[-1:] + [21, 22]
        response_logprobs = [0.0] * len(response_ids)
        return response_ids, response_logprobs, False


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del messages, tools, add_generation_prompt, tokenize, kwargs
        # Minimal tokenization: return a small prompt.
        return [101, 102]

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return "<decoded>"


def _pad_1d(ids: list[int], *, length: int, pad_id: int = 0) -> list[int]:
    if len(ids) > length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def _to_internal(
    *,
    output_prompt_ids: list[int],
    output_response_ids: list[int],
    output_response_mask: list[int],
    metrics: AgentLoopMetrics,
    extra_fields: dict[str, Any],
    num_turns: int,
    prompt_len: int,
    response_len: int,
) -> _InternalAgentLoopOutput:
    prompt_ids = _pad_1d(output_prompt_ids, length=prompt_len, pad_id=0)
    response_ids = _pad_1d(output_response_ids, length=response_len, pad_id=0)
    response_mask = _pad_1d(output_response_mask, length=response_len, pad_id=0)

    seq_len = prompt_len + response_len
    attention_mask = _pad_1d([1] * len(output_prompt_ids), length=prompt_len, pad_id=0) + _pad_1d(
        [1] * len(output_response_ids),
        length=response_len,
        pad_id=0,
    )
    input_ids = prompt_ids + response_ids
    position_ids = list(range(seq_len))

    def t(x: list[int]) -> torch.Tensor:
        return torch.tensor([x], dtype=torch.long)

    return _InternalAgentLoopOutput(
        prompt_ids=t(prompt_ids),
        response_ids=t(response_ids),
        response_mask=t(response_mask),
        attention_mask=t(attention_mask),
        input_ids=t(input_ids),
        position_ids=t(position_ids),
        response_logprobs=None,
        routed_experts=None,
        multi_modal_inputs=None,
        multi_modal_data=None,
        reward_score=None,
        num_turns=num_turns,
        metrics=metrics,
        extra_fields=extra_fields,
    )


@pytest.mark.asyncio
async def test_agent_loop_extra_fields_schema_stable_for_training_concat_on_cpu():
    # Minimal config surface used by the agent loops.
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 16, "response_length": 16, "multi_turn": {"tool_config_path": None}},
                "model": {},
            },
            "data": {
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
            },
        }
    )

    server_manager = _FakeServerManager()
    tokenizer = _FakeTokenizer()
    processor = None

    trainer_config = DictConfigWrap(config)
    data_config = DictConfigWrap(config.data)

    single_turn = SingleTurnAgentLoop(
        trainer_config=trainer_config,
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=RLHFDataset,
        data_config=data_config,
    )

    raw_prompt = [{"role": "user", "content": "hi"}]
    sampling_params: dict[str, Any] = {}

    out = await single_turn.run(sampling_params=sampling_params, raw_prompt=raw_prompt)

    # Agent loop outputs should always contain these fields with consistent types.
    assert out.extra_fields["turn_scores"] == []
    assert out.extra_fields["tool_rewards"] == []

    internal_a = _to_internal(
        output_prompt_ids=out.prompt_ids,
        output_response_ids=out.response_ids,
        output_response_mask=out.response_mask,
        metrics=out.metrics,
        extra_fields=out.extra_fields,
        num_turns=out.num_turns,
        prompt_len=len(out.prompt_ids),
        response_len=len(out.response_ids),
    )

    # Mimic two "worker chunks" and concatenate as in training.
    dummy_worker = type("_DummyWorker", (), {"reward_loop_worker_handles": None})()
    merged = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[internal_a],
        input_non_tensor_batch={
            "index": np.array([0], dtype=object),
            "agent_name": np.array(["single_turn_agent"], dtype=object),
        },
    )

    # Stable schema: present regardless of which loop produced a sample.
    stable_keys = (
        "turn_scores",
        "tool_rewards",
        "min_global_steps",
        "max_global_steps",
        "extras",
    )
    for key in stable_keys:
        assert key in merged.non_tensor_batch, f"missing key in merged batch: {key}"
        assert merged.non_tensor_batch[key].shape == (1,), (
            f"invalid shape for {key}: {merged.non_tensor_batch[key].shape}"
        )

    # And the list-typed fields are actually lists (not missing / scalar).
    assert merged.non_tensor_batch["turn_scores"][0] == []
    assert merged.non_tensor_batch["tool_rewards"][0] == []
