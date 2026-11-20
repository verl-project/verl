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
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
    _align_generated_turn_spans,
)
from verl.trainer.ppo.process_reward import ProcessRewardRunner, generated_turn_spans
from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder, MergeResult


def make_data():
    data = AgentData([], None, None, None, None, {}, "test", {})
    data.prompt_ids = [100]
    return data


def apply_merge(data, merge):
    # Use the same builder alignment that drives ToolAgentLoop.
    mask, _ = ContinuousTokenBuilder(None).align_response_metadata(merge, data.response_mask)
    _align_generated_turn_spans(data, merge)
    data.prompt_ids = merge.token_ids
    data.response_mask = mask
    assert generated_turn_spans(mask, data.generated_turn_spans) == data.generated_turn_spans


def test_continuous_token_edits_preserve_adjacent_turns_and_skip_boundaries():
    data = make_data()
    apply_merge(data, MergeResult([100, 1, 2], 2, kind="assistant"))
    apply_merge(data, MergeResult([100, 1, 2, 3], 1, kind="assistant"))
    assert data.generated_turn_spans == [(0, 2), (2, 3)]
    # A model-specific context merge trims one generated suffix token, then inserts a boundary.
    apply_merge(
        data, MergeResult([100, 1, 2, 90, 20], 1, kind="context", inserted_token_ids=[90], removed_prefix_token_count=1)
    )
    assert data.generated_turn_spans == [(0, 2)]
    apply_merge(data, MergeResult([100, 1, 2, 90, 20, 91, 4, 5], 2, kind="assistant", inserted_token_ids=[91]))
    assert data.response_mask == [1, 1, 0, 0, 0, 1, 1]
    assert data.generated_turn_spans == [(0, 2), (5, 7)]


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", [0, 1, 2, 3])
async def test_context_suffix_removal_keeps_rewards_aligned_with_retained_turns(removed):
    data = make_data()
    data.use_process_reward = True
    apply_merge(data, MergeResult([100, 1, 2], 2, kind="assistant"))
    apply_merge(data, MergeResult([100, 1, 2, 3], 1, kind="assistant"))
    data.extra_fields["turn_rewards"] = [-1.0, 0.5]
    retained = 3 - removed
    apply_merge(
        data,
        MergeResult([100, *[1, 2, 3][:retained], 20], 1, kind="context", removed_prefix_token_count=removed),
    )
    rewards = [-1.0, 0.5] if retained == 3 else [-1.0] if retained else []
    assert data.extra_fields["turn_rewards"] == rewards
    output = SimpleNamespace(
        response_ids=data.prompt_ids[1:],
        response_mask=data.response_mask,
        extra_fields={**data.extra_fields, "generated_turn_spans": data.generated_turn_spans},
    )
    scores = await ProcessRewardRunner([]).score(output, {})
    torch.testing.assert_close(scores, torch.tensor([*[-1.0, -1.0, 0.5][:retained], 0.0]))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("feedback", [None, [1.0], [float("nan"), 0.5]])
def test_context_suffix_removal_validates_feedback_only_when_enabled(enabled, feedback):
    data = make_data()
    data.use_process_reward = enabled
    data.prompt_ids = [100, 1, 2, 3]
    data.response_mask = [1, 1, 1]
    data.generated_turn_spans = [(0, 2), (2, 3)]
    data.extra_fields["turn_rewards"] = feedback
    merge = MergeResult([100, 1, 2, 20], 1, kind="context", removed_prefix_token_count=1)
    if enabled:
        with pytest.raises(ValueError, match="turn_rewards"):
            apply_merge(data, merge)
    else:
        apply_merge(data, merge)
        assert data.extra_fields["turn_rewards"] is feedback


@pytest.mark.asyncio
async def test_generating_state_records_actual_continuous_token_turns():
    builder = ContinuousTokenBuilder(None)
    data = make_data()

    async def generate(**kwargs):
        return SimpleNamespace(token_ids=[11, 12], log_probs=[], num_preempted=0, routed_experts=None, extra_fields={})

    async def merge(prompt, response, mask, logprobs, **kwargs):
        result = builder.merge_assistant_tokens(prompt, response)
        aligned, _ = builder.align_response_metadata(result, mask)
        return result, aligned, None

    async def parse(*args):
        return "answer", []

    loop = SimpleNamespace(
        tool_parser=SimpleNamespace(stop_token_ids=[], extract_tool_calls=parse),
        server_manager=SimpleNamespace(generate=generate),
        ct_merge_assistant_token=merge,
        response_length=100,
        max_assistant_turns=0,
        max_user_turns=0,
        tools={},
        _build_assistant_message=lambda content, data: {"role": "assistant", "content": content},
    )
    for _ in range(2):
        assert await ToolAgentLoop._handle_generating_state(loop, data, {}) == AgentState.TERMINATED
    assert data.generated_turn_spans == [(0, 2), (2, 4)]
    assert data.response_mask == [1, 1, 1, 1]


async def run_tool_loop(feedback, *, limit=10, enabled=True, validate=False):
    async def info(messages):
        return {}

    async def pending(data, params):
        # An already-generated trajectory supplied by a fake generation stage.
        data.prompt_ids = [100, 1, 2, 20, 21, 3, 4]
        data.response_mask = [1, 1, 0, 0, 1, 1]
        data.generated_turn_spans = [(0, 2), (4, 6)]
        data.extra_fields["turn_rewards"] = feedback
        return AgentState.TERMINATED

    loop = SimpleNamespace(
        config=OmegaConf.create({"algorithm": {"process_reward": {"enabled": enabled}}}),
        process_multi_modal_info=info,
        _get_mm_processor_kwargs=lambda audios: {},
        _make_request_id=ToolAgentLoop._make_request_id,
        rollout_config=SimpleNamespace(full_determinism=False),
        tools={},
        tool_schemas=[],
        _handle_pending_state=pending,
        response_length=limit,
    )
    return await ToolAgentLoop.run(loop, {}, validate=validate, raw_prompt=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,expected", [(1, [-1.0]), (3, [-1.0]), (5, [-1.0, 0.5]), (10, [-1.0, 0.5])])
async def test_tool_loop_finalization_clips_rewards_and_spans_together(limit, expected):
    output = await run_tool_loop([-1.0, 0.5], limit=limit)
    assert output.extra_fields["turn_rewards"] == expected
    scores = await ProcessRewardRunner([]).score(output, {})
    torch.testing.assert_close(scores, torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.5, 0.5][:limit]))


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,validate", [(False, False), (False, True), (True, True), (True, False)])
@pytest.mark.parametrize("feedback", [None, [1.0], [float("nan"), 0.5]])
async def test_tool_loop_strict_feedback_validation_is_opt_in_and_training_only(enabled, validate, feedback):
    if enabled and not validate:
        with pytest.raises(ValueError, match="turn_rewards"):
            await run_tool_loop(feedback, enabled=enabled, validate=validate, limit=1)
    else:
        output = await run_tool_loop(feedback, enabled=enabled, validate=validate, limit=1)
        assert output.extra_fields["turn_rewards"] == feedback


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,validate", [(False, False), (False, True), (True, True)])
async def test_tool_loop_bypass_preserves_untruncated_feedback(enabled, validate):
    feedback = [-1.0, 0.5]
    output = await run_tool_loop(feedback, enabled=enabled, validate=validate, limit=1)
    assert output.extra_fields["turn_rewards"] == feedback
