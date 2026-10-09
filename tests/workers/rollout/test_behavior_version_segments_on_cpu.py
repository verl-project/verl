# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Token provenance across actual async-client abort/resume iterations."""

import asyncio
from types import SimpleNamespace

import pytest

from verl.utils.metric.behavior_age import behavior_age_metrics
from verl.workers.rollout import llm_server
from verl.workers.rollout.replica import TokenOutput


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_real_client_tracks_only_new_tokens_across_versions(monkeypatch, enabled):
    outputs = iter(
        [
            TokenOutput(
                token_ids=[3, 4], log_probs=[-1.0, -2.0], stop_reason="aborted", extra_fields={"global_steps": 0}
            ),
            TokenOutput(token_ids=[], log_probs=[], stop_reason="abort", extra_fields={"global_steps": 1}),
            TokenOutput(token_ids=[5], log_probs=[-3.0], stop_reason="stop", extra_fields={"global_steps": 2}),
        ]
    )
    prompts = []
    budgets = []

    async def generate(self, **kwargs):
        prompts.append(kwargs["prompt_ids"])
        budgets.append(kwargs["sampling_params"]["max_tokens"])
        return next(outputs)

    async def no_wait(*args):
        pass

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(collect_behavior_version_metrics=enabled, response_length=5)
        )
    )
    client = llm_server.FullyAsyncLLMServerClient(config=config, load_balancer_handle=None)
    output = await client.generate("request", prompt_ids=[1, 2], sampling_params={"max_tokens": 5})
    assert output.token_ids == [3, 4, 5]
    assert output.log_probs == [-1.0, -2.0, -3.0]
    assert prompts == [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4]]
    assert budgets == [5, 3, 3]
    if enabled:
        assert output.extra_fields["behavior_version_segments"] == [[0, 2], [2, 1]]
        metrics = behavior_age_metrics([output.extra_fields], [3], [True], 2)
        assert metrics["training/off_policy/token_staleness/mean"] == pytest.approx(4 / 3)
        assert metrics["training/off_policy/behavior_version/cross_version_response_fraction"] == 1
    else:
        assert "behavior_version_segments" not in output.extra_fields


@pytest.mark.asyncio
async def test_missing_server_version_reduces_coverage(monkeypatch):
    async def generate(self, **kwargs):
        return TokenOutput(token_ids=[3], log_probs=None, stop_reason="stop")

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(collect_behavior_version_metrics=True))
    )
    client = llm_server.FullyAsyncLLMServerClient(config=config, load_balancer_handle=None)
    output = await client.generate("request", prompt_ids=[1], sampling_params={"max_tokens": 2})
    result = behavior_age_metrics([output.extra_fields], [1], [True], 2)
    assert result["training/off_policy/behavior_version/response_coverage"] == 0
    assert not any("token_staleness" in key for key in result)


@pytest.mark.asyncio
@pytest.mark.parametrize("segments", [None, [], [[7, 2]], [[None, 2]]])
async def test_existing_partial_provenance_continues_when_collection_is_disabled(monkeypatch, segments):
    extra_fields = {} if segments is None else {"behavior_version_segments": segments}
    restored = TokenOutput(token_ids=[3, 4], log_probs=[-1.0, -2.0], num_preempted=0, extra_fields=extra_fields)
    outputs = iter(
        [
            TokenOutput(token_ids=[], log_probs=[], stop_reason="abort", extra_fields={"global_steps": 8}),
            TokenOutput(token_ids=[5], log_probs=[-3.0], stop_reason="stop", extra_fields={"global_steps": 9}),
        ]
    )
    prompts = []

    async def generate(self, **kwargs):
        prompts.append(kwargs["prompt_ids"])
        return next(outputs)

    async def no_wait(*args):
        pass

    # Model the generic restored TokenOutput insertion point without requiring
    # checkpoint persistence in this independently usable PR.
    monkeypatch.setattr(llm_server, "TokenOutput", lambda **kwargs: restored)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(collect_behavior_version_metrics=False))
    )
    client = llm_server.FullyAsyncLLMServerClient(config=config, load_balancer_handle=None)
    output = await client.generate("request", prompt_ids=[1, 2], sampling_params={"max_tokens": 5})

    assert output.token_ids == [3, 4, 5]
    assert output.log_probs == [-1.0, -2.0, -3.0]
    assert prompts == [[1, 2, 3, 4], [1, 2, 3, 4]]
    if segments:
        assert output.extra_fields["behavior_version_segments"] == [segments[0], [9, 1]]
        metrics = behavior_age_metrics([output.extra_fields], [3], [True], current_version=9)
        if segments[0][0] is None:
            assert metrics["training/off_policy/behavior_version/token_coverage"] == 0
            assert "training/off_policy/token_staleness/mean" not in metrics
        else:
            assert metrics["training/off_policy/token_staleness/mean"] == pytest.approx(4 / 3)
    elif segments is None:
        assert "behavior_version_segments" not in output.extra_fields
    else:
        assert output.extra_fields["behavior_version_segments"] == []
