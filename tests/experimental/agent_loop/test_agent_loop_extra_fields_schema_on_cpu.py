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
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import verl.experimental.agent_loop.agent_loop as agent_loop_module
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.schemas import MODEL_ROW_ROUTE_SEMANTICS


class _FakeServerManager:
    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        del request_id, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs, kwargs
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
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[list[int], list[float], bool]:
        del request_id, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs
        # Return a short partial generation and "not cancelled".
        response_ids = prompt_ids[-1:] + [21, 22]
        response_logprobs = [0.0] * len(response_ids)
        return response_ids, response_logprobs, False


class _FakeTokenizer:
    padding_side = "right"

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
        encoded_inputs: dict[str, list[int]],
        *,
        padding: str,
        max_length: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
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


def _make_agent_loop_output() -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
    )


def test_agent_loop_output_as_dict_handles_default_extra_fields():
    fields = _make_agent_loop_output().as_dict()

    assert fields["extra_fields"] == {}


def test_agent_loop_output_as_dict_preserves_mutated_default_extra_fields():
    output = _make_agent_loop_output()
    output.extra_fields["raw_prompt"] = [{"role": "user", "content": "hello"}]

    fields = output.as_dict()

    assert fields["extra_fields"] == {"raw_prompt": [{"role": "user", "content": "hello"}]}


def test_agent_loop_output_as_dict_promotes_teacher_fields_without_mutating_model():
    output = _make_agent_loop_output()
    output.extra_fields.update(
        {
            "raw_prompt": [{"role": "user", "content": "hello"}],
            "teacher_ids": [201, 202],
            "teacher_logprobs": [-0.1, -0.2],
        }
    )

    fields = output.as_dict()

    assert fields["teacher_ids"] == [201, 202]
    assert fields["teacher_logprobs"] == [-0.1, -0.2]
    assert fields["extra_fields"] == {"raw_prompt": [{"role": "user", "content": "hello"}]}
    assert output.extra_fields["teacher_ids"] == [201, 202]
    assert output.extra_fields["teacher_logprobs"] == [-0.1, -0.2]


def test_agent_loop_output_as_dict_promotes_rollout_topk_fields():
    output = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": [[4, 9], [5, 8]],
            "response_topk_log_probs": [[-0.1, -2.0], [-0.2, -1.5]],
        },
    )

    fields = output.as_dict()

    # Unpadded full-sequence layout: row P-1+r holds response token r's head, other rows the dummy head.
    assert fields["rollout_topk_ids"].shape == (5, 2) and fields["rollout_topk_ids"].dtype == torch.int32
    assert fields["rollout_topk_log_probs"].shape == (5, 2) and fields["rollout_topk_log_probs"].dtype == torch.float32
    assert fields["rollout_topk_ids"][2].tolist() == [4, 9] and fields["rollout_topk_ids"][3].tolist() == [5, 8]
    assert fields["rollout_topk_ids"][0].tolist() == [0, 1]
    assert fields["rollout_support_complete"].tolist() == [True, True, False, False, True]
    assert fields["extra_fields"] == {}
    assert output.extra_fields["response_topk_ids"] == [[4, 9], [5, 8]]


@pytest.mark.parametrize(
    "response_complete",
    [[True, False], np.array([True, False], dtype=np.bool_), torch.tensor([True, False])],
    ids=["bool-list", "numpy-bool", "torch-bool"],
)
def test_agent_loop_output_aligns_provider_support_evidence_with_prediction_rows(response_complete):
    output = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": [[4, 9], [5, 8]],
            "response_topk_log_probs": [[0.0, float("-inf")], [-0.2, -1.5]],
            "response_support_complete": response_complete,
        },
    )
    fields = output.as_dict()
    assert fields["rollout_support_complete"].dtype == torch.bool
    assert fields["rollout_support_complete"].tolist() == [True, True, True, False, True]
    assert fields["extra_fields"] == {}


@pytest.mark.parametrize(
    "response_complete",
    [[1, 0], np.array([0.25, 0.0], dtype=np.float32), [float("nan"), 0.0], [True, 1], [True, 0.25]],
    ids=["integers", "floating-array", "nan", "mixed-bool-int", "mixed-bool-float"],
)
def test_agent_loop_output_rejects_nonboolean_provider_support_evidence(response_complete):
    output = _make_agent_loop_output()
    output.extra_fields.update(
        {
            "response_topk_ids": [[11, 13], [12, 14]],
            "response_topk_log_probs": [[0.0, float("-inf")], [-0.2, -1.5]],
            "response_support_complete": response_complete,
        }
    )

    with pytest.raises(ValueError, match="one bool per response token"):
        output.as_dict()


def test_agent_loop_output_rejects_incomplete_provider_evidence_records():
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": [[3, 9], [4, 8]],
            "response_topk_log_probs": [[-0.1, -2.0], [-0.2, -1.5]],
            "response_support_complete": [True],
        },
    )
    with pytest.raises(ValueError, match="one bool per response token"):
        output.as_dict()


def test_agent_loop_output_as_dict_rejects_multi_turn_rollout_topk():
    output = AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        num_turns=4,
        metrics=AgentLoopMetrics(),
        extra_fields={"response_topk_ids": [[2, 9], [3, 8]], "response_topk_log_probs": [[-0.1, -2.0], [-0.2, -1.5]]},
    )

    with pytest.raises(ValueError, match="single-turn"):
        output.as_dict()


def test_agent_loop_output_as_dict_rejects_misaligned_rollout_topk():
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": np.array([[3, 9], [4, 8], [5, 7]], dtype=np.int32),
            "response_topk_log_probs": np.full((3, 2), -0.5, dtype=np.float32),
        },
    )

    with pytest.raises(ValueError, match="cover 3 tokens"):
        output.as_dict()


@pytest.mark.parametrize(
    "rows,mask,message",
    [
        (2, [1, 1], "shorter"),
        (3, [0, 1], "single-turn"),
        (3, [1], "single-turn"),
        (None, [1, 1], "expert IDs"),
        (0, [1, 1], "shorter"),
        ("rank", [1, 1], "layers"),
    ],
)
def test_independent_serialization_checks_natural_capture_rows(rows, mask, message):
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=mask,
        metrics=AgentLoopMetrics(),
        routed_experts=(None if rows is None else np.zeros((3, 1) if rows == "rank" else (rows, 1, 1), dtype=np.int32)),
        routed_experts_source_semantics=MODEL_ROW_ROUTE_SEMANTICS,
    )
    with pytest.raises(ValueError, match=message):
        output.as_dict()


@pytest.mark.asyncio
@pytest.mark.parametrize("route_case", ["plain", "model_rows", "malformed_shape"])
@pytest.mark.parametrize("trajectory_list", [False, True])
async def test_agent_loop_shared_exit_validates_routes_and_passes_scalar_model_type(
    monkeypatch, route_case, trajectory_list
):
    captured_kwargs: dict[str, Any] = {}
    sample = AgentLoopOutput(prompt_ids=[101, 102], response_ids=[11], response_mask=[1], metrics=AgentLoopMetrics())
    if route_case != "plain":
        routes = np.array([[[256, 895]], [[0, 1]]], dtype=np.int32)
        if route_case == "malformed_shape":
            routes = routes[:, :, 0]
        routes.setflags(write=False)
        sample.routed_experts = routes
        sample.routed_experts_source_semantics = MODEL_ROW_ROUTE_SEMANTICS
    expected_output = [sample] if trajectory_list else sample

    class _FakeAgentLoop:
        async def run(self, sampling_params: dict[str, Any], **kwargs):
            del sampling_params, kwargs
            return expected_output

    def fake_instantiate(*, config, **kwargs):
        del config
        captured_kwargs.update(kwargs)
        return _FakeAgentLoop()

    monkeypatch.setattr(agent_loop_module.hydra.utils, "instantiate", fake_instantiate)
    monkeypatch.setattr(agent_loop_module, "rollout_trace_attr", lambda **kwargs: nullcontext())
    monkeypatch.setitem(agent_loop_module._agent_loop_registry, "scalar_model_type_test", {"_target_": "unused"})

    worker = object.__new__(AgentLoopWorker)
    worker.config = OmegaConf.create({"data": {}})
    worker.llm_client = object()
    worker.tokenizer = _FakeTokenizer()
    worker.processor = None
    worker.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_moe"))
    worker.hf_model_type = "qwen2_5_vl"
    worker.dataset_cls = RLHFDataset
    worker.tools = []

    async def passthrough_postprocess(output, validate, **kwargs):
        del validate, kwargs
        return output

    worker._agent_loop_postprocess = passthrough_postprocess

    async def run():
        return await worker._run_agent_loop(
            {},
            {"step": 0, "sample_index": 0, "rollout_n": 0, "validate": False},
            agent_name="scalar_model_type_test",
        )

    if route_case == "malformed_shape":
        with pytest.raises(ValueError, match="shape|layers"):
            await run()
    else:
        assert await run() is expected_output
        if route_case == "model_rows":
            assert isinstance(sample.routed_experts, torch.Tensor)
    assert captured_kwargs["hf_model_type"] == "qwen2_5_vl"
    assert isinstance(captured_kwargs["hf_model_type"], str)
    assert "hf_config" not in captured_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_name", ["legacy", "transfer_queue"])
@pytest.mark.parametrize("capture", ["valid", "malformed_shape", "tool_mask", "short", "missing_ids"])
async def test_model_row_validation_precedes_both_real_exits(monkeypatch, exit_name, capture):
    import importlib.util
    import sys
    from pathlib import Path
    from types import MethodType, ModuleType
    from unittest.mock import AsyncMock, patch

    routes = torch.tensor([[[256, 895]], [[0, 1]]], dtype=torch.int32)
    sample = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11],
        response_mask=[1],
        routed_experts=routes,
        routed_experts_source_semantics=MODEL_ROW_ROUTE_SEMANTICS,
        metrics=AgentLoopMetrics(),
    )
    error = None
    if capture == "malformed_shape":
        sample.routed_experts = routes[:, :, 0]
        error = "shape|layers"
    elif capture == "tool_mask":
        sample.response_mask = [0]
        error = "single-turn"
    elif capture == "short":
        sample.routed_experts = routes[:1]
        error = "shorter"
    elif capture == "missing_ids":
        sample.routed_experts = None
        error = "expert IDs"

    class Loop:
        async def run(self, *args, **kwargs):
            return sample

    monkeypatch.setattr(agent_loop_module.hydra.utils, "instantiate", lambda **kwargs: Loop())
    monkeypatch.setattr(agent_loop_module, "rollout_trace_attr", lambda **kwargs: nullcontext())
    monkeypatch.setitem(agent_loop_module._agent_loop_registry, "test_kimi_exits", {})
    worker = object.__new__(AgentLoopWorker)
    worker.config = OmegaConf.create({"data": {}})
    worker.llm_client = None
    worker.tokenizer = _FakeTokenizer()
    worker.processor = None
    worker.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_moe"))
    worker.hf_model_type = "kimi_k3"
    worker.dataset_cls = RLHFDataset
    worker.tools = []
    worker.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 4})
    worker.mm_processor_kwargs = {}
    worker.reward_loop_worker_handles = None
    worker.distillation_enabled = False
    put = AsyncMock()
    if exit_name == "transfer_queue":
        # Isolate the external service; import and run the actual decorated worker.
        tq = ModuleType("transfer_queue")
        tq.async_kv_batch_put = put
        path = Path(agent_loop_module.__file__).parents[2] / "trainer/ppo/v1/agent_loop_tq.py"
        spec = importlib.util.spec_from_file_location("agent_loop_tq_cpu", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"transfer_queue": tq}):
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        cls = module.AgentLoopWorkerTQ.__ray_metadata__.modified_class
        worker._agent_loop_postprocess = MethodType(cls._agent_loop_postprocess, worker)

    async def run():
        return await worker._run_agent_loop(
            {},
            {"step": 0, "sample_index": 0, "rollout_n": 0, "validate": False},
            agent_name="test_kimi_exits",
            raw_prompt=[],
            uid="prompt",
            session_id=0,
            global_steps=0,
        )

    if error:
        with pytest.raises(ValueError, match=error):
            await run()
        put.assert_not_called()
    else:
        result = await run()
        if exit_name == "transfer_queue":
            fields = put.call_args.kwargs["fields"]
            actual_ids = fields["routed_experts"]
            actual_ids = actual_ids.values() if actual_ids.is_nested else actual_ids[0]
            assert list(fields["routed_experts_transport_semantics"]) == [MODEL_ROW_ROUTE_SEMANTICS]
        else:
            actual_ids = result.routed_experts.values()
            # Exercise the real legacy batch exit, not just per-sample padding.
            batch_output = worker._postprocess([result])
            from verl.utils import tensordict_utils as tu

            assert (
                tu.get_non_tensor_data(batch_output.batch, "routed_experts_transport_semantics", None)
                == MODEL_ROW_ROUTE_SEMANTICS
            )
            torch.testing.assert_close(batch_output.batch["routed_experts"].values(), routes)
        torch.testing.assert_close(actual_ids, routes)


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
@pytest.mark.parametrize("model_rows", [False, True])
async def test_agent_loop_extra_fields_schema_stable_for_training_concat_on_cpu(model_rows):
    # Minimal config surface used by the agent loops.
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 16, "response_length": 16, "multi_turn": {"tool_config_path": None}},
                # AgentLoopBase unconditionally builds the Continuous Token builder; without an
                # HF model type or multimodal processor it falls back to the default builder.
                "model": {"path": "dummy-model", "tokenizer_path": "dummy-model"},
            },
            "data": {
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
            },
        }
    )

    server_manager = _FakeServerManager()
    if model_rows:
        base_generate = server_manager.generate

        async def generate_with_model_rows(*args, **kwargs):
            output = await base_generate(*args, **kwargs)
            output.routed_experts = np.tile(np.array([[[256, 895]]], dtype=np.int32), (50, 1, 1))
            output.routed_experts_source_semantics = MODEL_ROW_ROUTE_SEMANTICS
            return output

        server_manager.generate = generate_with_model_rows
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

    if model_rows:
        # Coordinate ownership comes from the producer tag, even without a Kimi model config.
        assert out.routed_experts.shape == (50, 1, 2)
        assert out.routed_experts_source_semantics == MODEL_ROW_ROUTE_SEMANTICS

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
@pytest.mark.parametrize("model_rows", [False, True])
async def test_agent_loop_postprocess_accepts_read_only_routed_experts_on_cpu(model_rows):
    class _DummyWorker:
        _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
        _compute_position_ids = AgentLoopWorker._compute_position_ids
        _get_mm_processor_kwargs = AgentLoopWorker._get_mm_processor_kwargs
        _compute_score = AgentLoopWorker._compute_score
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        _pad_token_ids = AgentLoopWorker._pad_token_ids
        distillation_enabled = False

        def __init__(self):
            self.tokenizer = _FakeTokenizer()
            self.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 4})
            self.processor = None
            self.mm_processor_kwargs = {}
            self.reward_loop_worker_handles = None

    worker = _DummyWorker()
    worker.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="kimi_k3", num_hidden_layers=2, num_experts=896, num_experts_per_token=1)
    )
    routed_experts = np.arange(8, dtype=np.int64).reshape(4, 2, 1)
    if model_rows:
        routed_experts = np.array([0, 255, 256, 895] * 2, dtype=np.int32).reshape(4, 2, 1)
    routed_experts.setflags(write=False)
    assert not routed_experts.flags.writeable

    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        routed_experts=routed_experts,
        routed_experts_source_semantics=MODEL_ROW_ROUTE_SEMANTICS if model_rows else None,
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
            worker,
            output,
            validate=False,
            raw_prompt=[{"role": "user", "content": "hi"}],
        )

    if model_rows:
        assert internal.routed_experts.is_nested
        expected = torch.tensor(routed_experts.copy())
        torch.testing.assert_close(internal.routed_experts.values(), expected)
        torch.testing.assert_close(output.as_dict()["routed_experts"], expected)
        return

    expected = torch.tensor(routed_experts.copy()).unsqueeze(0)
    assert internal.routed_experts is not None
    assert internal.routed_experts.dtype == expected.dtype
    assert internal.routed_experts.shape == (1, 8, 2, 1)
    torch.testing.assert_close(internal.routed_experts[:, 2:6], expected)
    assert torch.count_nonzero(internal.routed_experts[:, :2]) == 0
    assert torch.count_nonzero(internal.routed_experts[:, 6:]) == 0


class _FakeTokenizerCustomPad:
    """A minimal tokenizer with a non-zero pad_token_id for testing."""

    pad_token_id = 42
    padding_side = "right"

    def pad(
        self,
        encoded_inputs: dict[str, list[int]],
        *,
        padding: str,
        max_length: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        input_ids = encoded_inputs["input_ids"]
        pad_len = max_length - len(input_ids)
        padded_ids = input_ids + [0] * pad_len
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        output = {"input_ids": torch.tensor([padded_ids], dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
        return output


@pytest.mark.asyncio
async def test_agent_loop_pad_token_ids_empty_with_non_zero_pad_id():
    """Regression test: empty token list uses tokenizer's pad_token_id (not hardcoded 0)."""
    worker = type(
        "_DummyWorker",
        (),
        {
            "_pad_token_ids": AgentLoopWorker._pad_token_ids,
            "tokenizer": _FakeTokenizerCustomPad(),
            "processor": None,
            "mm_processor_kwargs": {},
        },
    )()

    result = worker._pad_token_ids(
        tokens=[],
        max_length=8,
        padding_side="right",
        return_attention_mask=True,
    )

    # input_ids should use the tokenizer's pad_token_id (42), not 0
    expected_input_ids = torch.full((1, 8), 42, dtype=torch.long)
    torch.testing.assert_close(result["input_ids"], expected_input_ids)

    # attention_mask should be all zeros
    expected_attention_mask = torch.zeros((1, 8), dtype=torch.long)
    torch.testing.assert_close(result["attention_mask"], expected_attention_mask)
