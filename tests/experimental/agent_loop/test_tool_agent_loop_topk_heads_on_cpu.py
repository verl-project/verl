# Copyright contributors to the PEAR project
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

"""ToolAgentLoop keeps one sampler top-k head per response token across its turns (score centering)."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse
from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder, MergeResult
from verl.workers.rollout.replica import TokenOutput

K = 2
PROMPT_IDS = [1, 2, 3]
BOUNDARY_ID, CONTEXT_IDS = 90, [31, 32]  # what a tool turn renders as: one boundary token, then its content


def _heads(token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """A distinct head per token: its id and one more, with log-probs that name the token."""
    ids = np.array([[token_id, token_id + 100] for token_id in token_ids], dtype=np.int32)
    return ids, np.array([[-token_id / 100, -9.0] for token_id in token_ids], dtype=np.float32)


class _Server:
    """Two assistant turns: three tokens whose provider reports coverage, then two without evidence."""

    def __init__(self, with_heads: bool):
        first_ids, first_log_probs = _heads([11, 12, 13])
        second_ids, second_log_probs = _heads([21, 22])
        first_fields = {
            "response_topk_ids": first_ids,
            "response_topk_log_probs": first_log_probs,
            "response_support_complete": np.array([True, False, True]),
        }
        second_fields = {"response_topk_ids": second_ids, "response_topk_log_probs": second_log_probs}
        self.outputs = [
            TokenOutput(
                token_ids=[11, 12, 13],
                log_probs=[-0.1, -0.2, -0.3],
                extra_fields={"max_global_steps": 4, **(first_fields if with_heads else {})},
            ),
            TokenOutput(
                token_ids=[21, 22],
                log_probs=[-0.4, -0.5],
                extra_fields={"max_global_steps": 5, **(second_fields if with_heads else {})},
            ),
        ]

    async def generate(self, **kwargs) -> TokenOutput:
        return self.outputs.pop(0)


class _ToolParser:
    """The first answer calls a tool, the second does not."""

    stop_token_ids = []

    def __init__(self):
        self.answers = [("", [FunctionCall(name="look", arguments="{}")]), ("done", [])]

    async def extract_tool_calls(self, response_ids, tools):
        return self.answers.pop(0)


def _tool_agent_loop(response_length: int, with_heads: bool = True) -> ToolAgentLoop:
    """A ToolAgentLoop on stand-ins for its server, its tokenizer and its tool."""
    builder = ContinuousTokenBuilder.__new__(ContinuousTokenBuilder)  # the alignments read no state of the builder

    async def no_multi_modal_info(messages):
        return {}

    async def ct_build_initial_tokens(messages, **kwargs):
        return list(PROMPT_IDS)

    async def ct_merge_assistant_token(prompt_ids, response_ids, response_mask, response_logprobs, **kwargs):
        merge = MergeResult(
            token_ids=list(prompt_ids) + list(response_ids), appended_token_count=len(response_ids), kind="assistant"
        )
        return merge, *builder.align_response_metadata(merge, response_mask, response_logprobs, **kwargs)

    async def ct_merge_context_msg(previous_messages, messages, prompt_ids, response_mask, response_logprobs, **kwargs):
        merge = MergeResult(
            token_ids=list(prompt_ids) + [BOUNDARY_ID] + CONTEXT_IDS,
            appended_token_count=len(CONTEXT_IDS),
            kind="context",
            inserted_token_ids=[BOUNDARY_ID],
        )
        return merge, *builder.align_response_metadata(merge, response_mask, response_logprobs)

    async def call_tool(tool_call, tools_kwargs, agent_data):
        return ToolResponse(text="seen"), None, {}

    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.rollout_config = SimpleNamespace(full_determinism=False)
    loop.tools, loop.tool_schemas = {}, []
    loop.tool_parser, loop.tool_parser_name = _ToolParser(), "stand_in"
    loop.server_manager = _Server(with_heads)
    loop.continuous_token_builder = builder
    loop.processor = None
    loop.response_length = response_length
    loop.max_assistant_turns = loop.max_user_turns = None
    loop.max_parallel_calls = 1
    loop.process_multi_modal_info = no_multi_modal_info
    loop._get_mm_processor_kwargs = lambda audios: {}
    loop._assert_mm_supported = lambda has_multi_modal: None
    loop.build_sglang_video_payload = lambda videos, mm_inputs: None
    loop.ct_build_initial_tokens = ct_build_initial_tokens
    loop.ct_merge_assistant_token = ct_merge_assistant_token
    loop.ct_merge_context_msg = ct_merge_context_msg
    loop._call_tool = call_tool
    return loop


def _run(loop: ToolAgentLoop, sampling_params: dict):
    return asyncio.run(loop.run(dict(sampling_params), raw_prompt=[{"role": "user", "content": "look around"}]))


def test_tool_agent_loop_keeps_one_head_per_response_token_across_turns():
    output = _run(_tool_agent_loop(response_length=64), {"topk_log_probs": K})

    # an assistant turn, a tool turn (boundary and content tokens), an assistant turn
    assert output.response_ids == [11, 12, 13, BOUNDARY_ID, *CONTEXT_IDS, 21, 22]
    assert output.response_mask == [1, 1, 1, 0, 0, 0, 1, 1]
    sampled = np.asarray(output.response_mask, dtype=bool)
    sampled_ids, sampled_log_probs = _heads([11, 12, 13, 21, 22])
    np.testing.assert_array_equal(output.extra_fields["response_topk_ids"][sampled], sampled_ids)
    np.testing.assert_array_equal(output.extra_fields["response_topk_log_probs"][sampled], sampled_log_probs)
    # the tokens the model did not sample carry the uniform placeholder head
    np.testing.assert_array_equal(output.extra_fields["response_topk_ids"][~sampled], [[0, 1]] * 3)
    np.testing.assert_allclose(output.extra_fields["response_topk_log_probs"][~sampled], -np.log(K))
    # the provider's coverage on the first turn, none on the second; a placeholder head is complete
    assert output.extra_fields["response_support_complete"].tolist() == [True, False, True, True, True, True] + [
        False,
        False,
    ]
    # the last turn's weight version, not the heads of the first turn
    assert output.extra_fields["max_global_steps"] == 5

    fields = output.as_dict()

    # response token r is predicted by row len(prompt) - 1 + r; the other rows hold the dummy head
    first_row = len(PROMPT_IDS) - 1
    assert fields["rollout_topk_ids"].shape == (len(PROMPT_IDS) + 8, K)
    assert torch.equal(
        fields["rollout_topk_ids"][first_row : first_row + 8],
        torch.from_numpy(output.extra_fields["response_topk_ids"]),
    )
    assert fields["rollout_support_complete"].tolist() == [True, True] + [True, False, True, True, True, True] + [
        False,
        False,
        True,
    ]
    assert "response_topk_ids" not in fields["extra_fields"]


def test_tool_agent_loop_cuts_the_heads_with_the_response_at_the_response_length():
    # the second assistant turn passes the response length: the episode ends there, cut to it
    output = _run(_tool_agent_loop(response_length=7), {"topk_log_probs": K})

    assert output.response_ids == [11, 12, 13, BOUNDARY_ID, *CONTEXT_IDS, 21]
    assert len(output.extra_fields["response_topk_ids"]) == len(output.extra_fields["response_topk_log_probs"]) == 7
    assert output.extra_fields["response_support_complete"].tolist() == [True, False, True, True, True, True, False]
    assert output.as_dict()["rollout_topk_ids"].shape == (len(PROMPT_IDS) + 7, K)


def test_tool_agent_loop_without_a_head_request_keeps_none():
    # a validation or greedy rollout: no topk_log_probs in the sampling parameters, no head from the server
    output = _run(_tool_agent_loop(response_length=64, with_heads=False), {})

    assert output.response_mask == [1, 1, 1, 0, 0, 0, 1, 1]
    assert "response_topk_ids" not in output.extra_fields
    assert "rollout_topk_ids" not in output.as_dict()


def test_tool_agent_loop_rejects_an_assistant_turn_without_its_heads():
    with pytest.raises(ValueError, match="assistant top-k heads are required"):
        _run(_tool_agent_loop(response_length=64, with_heads=False), {"topk_log_probs": K})
