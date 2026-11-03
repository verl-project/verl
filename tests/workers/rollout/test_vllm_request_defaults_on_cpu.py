# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Run the real server control flow with only vLLM/Ray endpoints isolated."""

from __future__ import annotations

import ast
import asyncio
import logging
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import (
    qwen2_5_vl_dedup_image_tokens,
    validate_topk_logprob_request,
    warn_topk_logprob_budget,
)


@pytest.fixture
def server_class():
    path = Path(__file__).resolve().parents[3] / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    cls = next(node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) == "vLLMHttpServer")
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    namespace = dict(
        asyncio=asyncio,
        uuid=uuid,
        logger=logging.getLogger(__name__),
        SamplingParams=lambda **values: SimpleNamespace(
            **(dict(prompt_logprobs=None, detokenize=True, stop=[], bad_words=[]) | values)
        ),
        TokensPrompt=dict,
        TokenOutput=TokenOutput,
        normalize_token_ids=normalize_token_ids,
        qwen2_5_vl_dedup_image_tokens=qwen2_5_vl_dedup_image_tokens,
        validate_topk_logprob_request=validate_topk_logprob_request,
        warn_topk_logprob_budget=warn_topk_logprob_budget,
        extract_prompt_logprobs=lambda **kwargs: None,
        RLInsightLogger=SimpleNamespace(
            trace_state=lambda *args, **kwargs: nullcontext(),
            trace_state_decorator=lambda *args, **kwargs: lambda method: method,
        ),
        STREAM_FINISHED=object(),
        ray=SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(get_actor_name=lambda: "test")),
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace["vLLMHttpServer"]


def _server(cls, *, custom=None, routes=None, aborted=False):
    server = object.__new__(cls)
    server.config = RolloutConfig(
        name="vllm",
        max_model_len=64,
        prompt_length=32,
        response_length=16,
        custom=custom,
        enable_rollout_routing_replay=routes is not None,
    )
    server.model_config = SimpleNamespace(processor=None, lora_rank=0, lora={})
    server.replica_rank = 0
    server.global_steps = 7
    server._disaggregation_role = "null"
    server._submission_paused = False
    server._admitting = 0
    server.requests = []

    async def add_request(request_id, prompt, sampling_params, **kwargs):
        server.requests.append(dict(request_id=request_id, prompt=prompt, sampling_params=sampling_params, **kwargs))
        output = SimpleNamespace(
            finished=True,
            outputs=[]
            if aborted
            else [
                SimpleNamespace(
                    token_ids=[9],
                    logprobs=None,
                    finish_reason="stop",
                    routed_experts=routes,
                    num_preempted=0,
                )
            ],
            num_cached_tokens=0,
            prompt_logprobs=None,
            kv_transfer_params={"remote_engine_id": "prefill"},
        )

        return SimpleNamespace(get_nowait=lambda: output, close=lambda: None)

    server.engine = SimpleNamespace(add_request=add_request)
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("row_semantics", [None, "model_rows"])
@pytest.mark.parametrize("ignore_eos", [False, True])
async def test_explicit_row_semantics_and_sampling_defaults_preserve_native_payload(
    server_class, row_semantics, ignore_eos
):
    routes = np.zeros((2, 1, 2), dtype=np.int32)
    custom = dict(sampling_defaults=dict(stop="END", max_tokens=5), routed_experts_source_semantics=row_semantics)
    server = _server(server_class, custom=custom, routes=routes)
    image = object()
    output = await server.generate([1, 2, 3], dict(ignore_eos=ignore_eos), "r", image_data=[image])
    request = server.requests[0]
    assert request["prompt"] == {"prompt_token_ids": [1, 2, 3], "multi_modal_data": {"image": [image]}}
    assert request["sampling_params"].max_tokens == 5
    assert request["sampling_params"].stop == "END" and request["sampling_params"].detokenize is True
    assert request["sampling_params"].ignore_eos is ignore_eos
    assert output.routed_experts is routes and output.routed_experts_source_semantics == row_semantics
    assert server._admitting == 0


@pytest.mark.asyncio
async def test_request_values_override_custom_defaults_and_keep_native_ignore_eos(server_class):
    server = _server(server_class, custom=dict(sampling_defaults=dict(stop="END", max_tokens=5, temperature=0.5)))
    await server.generate([1], dict(stop=[], max_tokens=2, temperature=0.7, ignore_eos=True), "r")
    params = server.requests[0]["sampling_params"]
    assert params.stop == [] and params.detokenize is False and params.ignore_eos is True
    assert params.max_tokens == 2 and params.temperature == 0.7


@pytest.mark.asyncio
async def test_ordinary_server_without_custom_config_keeps_legacy_routes_and_no_hf_dependency(server_class):
    routes = np.zeros((2, 1, 2), dtype=np.int32)
    server = _server(server_class, routes=routes)
    output = await server.generate([1, 2], {}, "r")
    assert output.routed_experts is routes and output.routed_experts_source_semantics is None
    assert server.requests[0]["sampling_params"].detokenize is False


@pytest.mark.asyncio
async def test_aborted_output_has_no_row_contract_without_ids(server_class):
    server = _server(server_class, custom=dict(routed_experts_source_semantics="model_rows"), aborted=True)
    output = await server.generate([1], {}, "r")
    assert output.stop_reason == "aborted" and output.routed_experts_source_semantics is None
    assert server._admitting == 0


@pytest.mark.asyncio
async def test_pd_legs_reuse_native_prompt_and_each_apply_request_defaults(server_class):
    custom = dict(sampling_defaults=dict(stop="END"))
    prefill, decode = _server(server_class, custom=custom), _server(server_class, custom=custom)
    prefill._disaggregation_role = "prefill"
    prefill._disaggregation_kv_transfer_config = {"kv_connector": "NixlConnector"}
    prefill._pd_decode_peers = [SimpleNamespace(generate=SimpleNamespace(remote=decode.generate))]
    prefill._pd_peer_idx = 0
    decode._disaggregation_role = "decode"
    await prefill.generate([1, 2, 3], {"max_tokens": 2}, "r", image_data=[object()])
    for server, suffix, max_tokens in [(prefill, "P", 1), (decode, "D", 2)]:
        assert len(server.requests) == 1
        request = server.requests[0]
        assert request["request_id"] == f"r_{suffix}"
        assert request["prompt"]["prompt_token_ids"] == [1, 2, 3]
        assert request["sampling_params"].max_tokens == max_tokens
        assert request["sampling_params"].stop == "END" and request["sampling_params"].detokenize is True


@pytest.mark.asyncio
@pytest.mark.parametrize("row_semantics", [None, "model_rows"])
async def test_tool_loop_preserves_explicit_row_contract_and_rejects_tool_mask(row_semantics):
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

    loop = object.__new__(ToolAgentLoop)
    loop.rollout_config = RolloutConfig(name="vllm")
    loop.response_length = 4
    loop.tools, loop.tool_schemas, loop.mm_processor_kwargs = {}, [], {}
    loop.max_assistant_turns, loop.max_user_turns = 1, None
    loop.tool_parser = SimpleNamespace(stop_token_ids=[])
    loop.process_multi_modal_info = AsyncMock(return_value={})
    routes = np.zeros((7 if row_semantics else 3, 1, 2), dtype=np.int32)
    loop.server_manager = SimpleNamespace(
        generate=AsyncMock(
            return_value=TokenOutput(
                token_ids=[8],
                routed_experts=routes,
                routed_experts_source_semantics=row_semantics,
            )
        )
    )

    async def pending(data, params):
        data.prompt_ids, data.response_mask = [1, 2, 3], [1, 0]
        return AgentState.GENERATING

    loop._handle_pending_state = pending
    loop.ct_merge_assistant_token = AsyncMock(return_value=(SimpleNamespace(token_ids=[1, 2, 3, 8]), [1, 0, 1], None))
    output = await ToolAgentLoop.run.__wrapped__(loop, {}, raw_prompt=[dict(role="user", content="q")])
    assert output.routed_experts_source_semantics == row_semantics
    if row_semantics:
        assert output.routed_experts.shape[0] == 7
        with pytest.raises(ValueError, match="single-turn"):
            output.as_dict()
    else:
        assert output.as_dict()["routed_experts"].shape == (4, 1, 2)
