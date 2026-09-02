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

from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse


def _result(invalid: bool | None) -> tuple[ToolResponse, float, dict]:
    metadata = {} if invalid is None else {"invalid_tool_call": invalid}
    return ToolResponse(text="tool result"), 0.0, metadata


def _tracking_state() -> SimpleNamespace:
    return SimpleNamespace(
        consecutive_invalid_tool_calls=0,
        max_consecutive_invalid_tool_calls_observed=0,
        invalid_tool_call_limit_reached=False,
        extra_fields={},
    )


def _track(limit: int | None, state: SimpleNamespace, results: list[tuple]) -> None:
    loop = SimpleNamespace(max_consecutive_invalid_tool_calls=limit)
    ToolAgentLoop._update_invalid_tool_call_tracking(loop, state, results)


def _write_diagnostics(limit: int | None, state: SimpleNamespace) -> None:
    loop = SimpleNamespace(max_consecutive_invalid_tool_calls=limit)
    ToolAgentLoop._write_invalid_tool_call_diagnostics(loop, state)


def test_disabled_limit_does_not_change_output_metadata() -> None:
    state = _tracking_state()

    _track(None, state, [_result(True), _result(True)])
    _write_diagnostics(None, state)

    assert state.extra_fields == {}
    assert state.invalid_tool_call_limit_reached is False


def test_enabled_limit_reports_zero_before_any_tool_result() -> None:
    state = _tracking_state()

    _track(2, state, [])
    _write_diagnostics(2, state)

    assert state.extra_fields == {
        "max_consecutive_invalid_tool_calls_observed": 0,
        "invalid_tool_call_limit_reached": False,
    }


def test_limit_is_reached_by_final_consecutive_invalid_call() -> None:
    state = _tracking_state()

    _track(2, state, [_result(True), _result(True)])
    _write_diagnostics(2, state)

    assert state.consecutive_invalid_tool_calls == 2
    assert state.max_consecutive_invalid_tool_calls_observed == 2
    assert state.invalid_tool_call_limit_reached is True
    assert state.extra_fields == {
        "max_consecutive_invalid_tool_calls_observed": 2,
        "invalid_tool_call_limit_reached": True,
        "termination_reason": "invalid_tool_call_limit",
    }


def test_valid_call_resets_the_streak() -> None:
    state = _tracking_state()

    _track(2, state, [_result(True), _result(False), _result(True)])

    assert state.consecutive_invalid_tool_calls == 1
    assert state.max_consecutive_invalid_tool_calls_observed == 1
    assert state.invalid_tool_call_limit_reached is False


def test_later_success_in_parallel_batch_cancels_limit_reached_mid_batch() -> None:
    state = _tracking_state()

    _track(2, state, [_result(True), _result(True), _result(False)])

    assert state.consecutive_invalid_tool_calls == 0
    assert state.max_consecutive_invalid_tool_calls_observed == 2
    assert state.invalid_tool_call_limit_reached is False


def test_unclassified_execution_error_leaves_streak_unchanged() -> None:
    state = _tracking_state()

    _track(2, state, [_result(True)])
    _track(2, state, [_result(None)])

    assert state.consecutive_invalid_tool_calls == 1
    assert state.invalid_tool_call_limit_reached is False


@pytest.mark.asyncio
async def test_processing_state_retains_limiting_response_at_response_length_boundary() -> None:
    async def call_tool(tool_call, tools_kwargs, agent_data):
        del tool_call, tools_kwargs, agent_data
        return ToolResponse(text="invalid call"), 0.0, {"invalid_tool_call": True}

    async def merge_tool_message(
        previous_messages,
        updated_messages,
        runtime_token_ids,
        response_mask,
        response_logprobs=None,
        *,
        tools=None,
    ):
        del previous_messages, updated_messages, response_logprobs, tools
        return SimpleNamespace(token_ids=[*runtime_token_ids, 41]), [*response_mask, 0], None

    loop = SimpleNamespace(
        max_parallel_calls=1,
        max_consecutive_invalid_tool_calls=1,
        processor=None,
        response_length=1,
        tool_schemas=[],
        _assert_mm_supported=lambda has_multi_modal: None,
        ct_merge_non_assistant_msg=merge_tool_message,
        _call_tool=call_tool,
    )
    loop._update_invalid_tool_call_tracking = ToolAgentLoop._update_invalid_tool_call_tracking.__get__(
        loop, ToolAgentLoop
    )
    agent_data = SimpleNamespace(
        messages=[{"role": "user", "content": "act"}],
        tool_calls=[FunctionCall(name="act", arguments="{}")],
        tools_kwargs={},
        metrics={},
        tool_rewards=[],
        prompt_ids=[1, 2, 3],
        response_mask=[],
        response_logprobs=[],
        image_data=None,
        user_turns=0,
        consecutive_invalid_tool_calls=0,
        max_consecutive_invalid_tool_calls_observed=0,
        invalid_tool_call_limit_reached=False,
        extra_fields={},
    )

    state = await ToolAgentLoop._handle_processing_tools_state(loop, agent_data)

    assert state is AgentState.TERMINATED
    assert agent_data.messages[-1] == {"role": "tool", "content": "invalid call"}
    assert agent_data.prompt_ids == [1, 2, 3, 41]
    assert agent_data.response_mask == [0]
    assert agent_data.user_turns == 1
    assert agent_data.invalid_tool_call_limit_reached is True
