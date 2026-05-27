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

import warnings
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.fully_async_policy.image_refs import INTERMEDIATE_TRAJECTORIES_KEY
from verl.experimental.fully_async_policy.intermediate_trajectory_utils import (
    expand_intermediate_trajectories_pre_log_prob,
)
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
    padding_side = "right"
    pad_token_id = 0

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

    def pad(
        self,
        encoded_inputs: dict[str, list[int]] | list[dict[str, list[int]]],
        *,
        padding: str,
        max_length: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        if isinstance(encoded_inputs, list):
            input_ids = encoded_inputs[0]["input_ids"]
        else:
            input_ids = encoded_inputs["input_ids"]
        if len(input_ids) > max_length:
            if self.padding_side == "left":
                input_ids = input_ids[-max_length:]
            else:
                input_ids = input_ids[:max_length]

        pad_len = max_length - len(input_ids)
        if self.padding_side == "left":
            padded_ids = [0] * pad_len + input_ids
            attention_mask = [0] * pad_len + [1] * len(input_ids)
        else:
            padded_ids = input_ids + [0] * pad_len
            attention_mask = [1] * len(input_ids) + [0] * pad_len

        output = {"input_ids": torch.tensor([padded_ids], dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
        return output

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return "<decoded>"


def _pad_1d(ids: list[int], *, length: int, pad_id: int = 0) -> list[int]:
    if len(ids) > length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def _object_array(values: list[Any]) -> np.ndarray:
    arr = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        arr[idx] = value
    return arr


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


class _FakeTeacherManager:
    def __init__(self, *, use_topk: bool = False, topk: int = 1):
        self.distillation_loss_config = SimpleNamespace(
            loss_settings=SimpleNamespace(use_topk=use_topk),
            topk=topk,
        )
        self.calls: list[tuple[list[int], Optional[str]]] = []

    async def compute_teacher_logprobs_single(
        self,
        *,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del multi_modal_data
        self.calls.append((list(sequence_ids), routing_key))
        k = self.distillation_loss_config.topk if self.distillation_loss_config.loss_settings.use_topk else 1
        teacher_ids = torch.tensor(
            [[token + offset for offset in range(k)] for token in sequence_ids],
            dtype=torch.int32,
        )
        teacher_logprobs = torch.full((len(sequence_ids), k), -0.5, dtype=torch.float32)
        return teacher_ids, teacher_logprobs


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
    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False},
    )()
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


@pytest.mark.asyncio
async def test_agent_loop_postprocess_accepts_read_only_routed_experts_on_cpu():
    class _DummyWorker:
        _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
        _compute_position_ids = AgentLoopWorker._compute_position_ids
        _compute_score = AgentLoopWorker._compute_score
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        distillation_enabled = False
        image_refs_enabled = False

        def __init__(self):
            self.tokenizer = _FakeTokenizer()
            self.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 4})
            self.processor = None
            self.reward_loop_worker_handles = None

    routed_experts = np.arange(8, dtype=np.int64).reshape(4, 2, 1)
    routed_experts.setflags(write=False)
    assert not routed_experts.flags.writeable

    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        routed_experts=routed_experts,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        internal = await AgentLoopWorker._agent_loop_postprocess(
            _DummyWorker(),
            output,
            validate=False,
            raw_prompt=[{"role": "user", "content": "hi"}],
        )

    expected = torch.tensor(routed_experts.copy()).unsqueeze(0)
    assert internal.routed_experts is not None
    assert internal.routed_experts.shape == (1, 8, 2, 1)
    torch.testing.assert_close(internal.routed_experts[:, 2:6], expected)
    assert torch.count_nonzero(internal.routed_experts[:, :2]) == 0
    assert torch.count_nonzero(internal.routed_experts[:, 6:]) == 0


@pytest.mark.asyncio
async def test_agent_loop_postprocess_adds_teacher_targets_to_intermediate_trajectories_on_cpu():
    class _DummyWorker:
        _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
        _compute_position_ids = AgentLoopWorker._compute_position_ids
        _compute_score = AgentLoopWorker._compute_score
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        _compute_intermediate_teacher_logprobs = AgentLoopWorker._compute_intermediate_teacher_logprobs
        _compute_teacher_logprobs_for_sequence = AgentLoopWorker._compute_teacher_logprobs_for_sequence
        _validate_teacher_logprob_shape = AgentLoopWorker._validate_teacher_logprob_shape
        distillation_enabled = True
        teacher_key = "data_source"
        image_refs_enabled = False

        def __init__(self):
            self.tokenizer = _FakeTokenizer()
            self.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 4})
            self.processor = None
            self.reward_loop_worker_handles = None
            self.teacher_server_manager = _FakeTeacherManager(use_topk=False)

    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={
            INTERMEDIATE_TRAJECTORIES_KEY: [
                {
                    "prompt_ids": [201, 202],
                    "response_ids": [21],
                    "response_mask": [1],
                    "metrics": AgentLoopMetrics().model_dump(),
                    "extra_fields": {},
                }
            ],
        },
    )

    worker = _DummyWorker()
    internal = await AgentLoopWorker._agent_loop_postprocess(
        worker,
        output,
        validate=False,
        raw_prompt=[{"role": "user", "content": "hi"}],
        data_source="math",
    )

    assert internal.teacher_ids is not None
    assert internal.teacher_logprobs is not None
    assert internal.teacher_ids.shape == (1, 8, 1)
    intermediate = internal.extra_fields[INTERMEDIATE_TRAJECTORIES_KEY][0]
    assert intermediate["extra_fields"]["teacher_ids"].shape == (3, 1)
    assert intermediate["extra_fields"]["teacher_logprobs"].shape == (3, 1)
    assert worker.teacher_server_manager.calls == [([101, 102, 11, 12], "math"), ([201, 202, 21], "math")]


@pytest.mark.parametrize("topk", [1, 3])
def test_expand_intermediate_trajectories_preserves_teacher_targets_on_cpu(topk: int):
    prompt_length = 4
    response_length = 4
    seq_len = prompt_length + response_length
    final_teacher_ids = torch.arange(seq_len * topk, dtype=torch.int32).view(1, seq_len, topk)
    final_teacher_logprobs = torch.full((1, seq_len, topk), -1.0, dtype=torch.float32)
    batch = TensorDict(
        {
            "prompts": torch.tensor([[0, 0, 101, 102]], dtype=torch.long),
            "responses": torch.tensor([[11, 12, 0, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.long),
            "input_ids": torch.tensor([[0, 0, 101, 102, 11, 12, 0, 0]], dtype=torch.long),
            "position_ids": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            "teacher_ids": final_teacher_ids,
            "teacher_logprobs": final_teacher_logprobs,
        },
        batch_size=1,
    )
    raw_teacher_ids = torch.arange(3 * topk, dtype=torch.int32).view(3, topk)
    raw_teacher_logprobs = torch.full((3, topk), -0.25, dtype=torch.float32)
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            INTERMEDIATE_TRAJECTORIES_KEY: _object_array(
                [
                    [
                        {
                            "prompt_ids": [201, 202],
                            "response_ids": [21],
                            "response_mask": [1],
                            "extra_fields": {
                                "teacher_ids": raw_teacher_ids,
                                "teacher_logprobs": raw_teacher_logprobs,
                            },
                        }
                    ]
                ]
            )
        },
    )

    expanded = expand_intermediate_trajectories_pre_log_prob(
        data,
        tokenizer=_FakeTokenizer(),
        processor=None,
        rollout_config=OmegaConf.create({"prompt_length": prompt_length, "response_length": response_length}),
        rollout_n=1,
    )

    assert len(expanded) == 2
    assert expanded.batch["teacher_ids"].shape == (2, seq_len, topk)
    assert expanded.batch["teacher_logprobs"].shape == (2, seq_len, topk)
    torch.testing.assert_close(expanded.batch["teacher_logprobs"][1, 2:5], raw_teacher_logprobs)
