# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
from types import SimpleNamespace

import pytest

from tests.workers.rollout.test_llm_server_routed_experts_on_cpu import _install_segments
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.termination import backend_termination


def _client():
    rollout = SimpleNamespace(collect_partial_rollout_metrics=True)
    return FullyAsyncLLMServerClient(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout)), load_balancer_handle=None
    )


def segment(tokens, native, control, stop=None):
    completion = None if native is None else SimpleNamespace(finish_reason=native, stop_reason=stop)
    return TokenOutput(
        token_ids=tokens,
        log_probs=[-1.0] * len(tokens),
        stop_reason=control,
        extra_fields={
            "global_steps": 0,
            "backend_termination": backend_termination(
                completion, max_tokens=2, unavailable_reason="empty_engine_output"
            ),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("native,stop", [("stop", 151645), ("length", None)])
async def test_native_stop_at_cap_is_not_relabelled_as_native_length(monkeypatch, native, stop):
    _install_segments(monkeypatch, [segment([1, 2], native, "completed", stop)])
    output = await _client().generate(request_id="r", prompt_ids=[10], sampling_params={"max_tokens": 2})
    assert output.stop_reason == "length"  # Existing cumulative budget control remains unchanged.
    term = output.extra_fields["termination"]
    assert term["response_budget_reached"] is True
    assert term["final_backend"]["finish_reason"] == native
    assert term["final_backend"]["stop_reason"] == stop
    assert output.token_ids == [1, 2] and output.log_probs == [-1.0, -1.0]


@pytest.mark.asyncio
async def test_abort_empty_abort_resume_preserves_terminal_source(monkeypatch):
    segments = [
        segment([1], "abort", "aborted"),
        segment([], None, "aborted"),
        segment([2], "stop", "completed", "end"),
    ]
    seen = _install_segments(monkeypatch, segments)
    output = await _client().generate(request_id="r", prompt_ids=[10], sampling_params={"max_tokens": 8})
    assert seen == [[10], [10, 1], [10, 1]]
    assert output.token_ids == [1, 2] and output.log_probs == [-1.0, -1.0]
    assert output.stop_reason == "completed"
    term = output.extra_fields["termination"]
    assert [a["backend_termination"]["finish_reason"] for a in term["attempts"]] == ["abort", None, "stop"]
    assert term["attempts"][1]["backend_termination"]["available"] is False
    assert term["final_backend"]["stop_reason"] == "end"
    assert term["response_budget_reached"] is False


@pytest.mark.asyncio
async def test_disabled_observation_does_not_read_attempt_metadata(monkeypatch):
    class UnobservedFields(dict):
        def get(self, key, default=None):
            if key in ("backend_termination", "engine_prefill"):
                pytest.fail("disabled observation read attempt metadata")
            return super().get(key, default)

    output = TokenOutput(token_ids=[1], log_probs=[-1.0], stop_reason="completed")
    output.extra_fields = UnobservedFields(global_steps=0)
    _install_segments(monkeypatch, [output])
    client = FullyAsyncLLMServerClient(config=SimpleNamespace(), load_balancer_handle=None)
    result = await client.generate(request_id="r", prompt_ids=[10], sampling_params={"max_tokens": 8})
    assert "termination" not in result.extra_fields
    assert "partial_rollout" not in result.extra_fields
    assert result.token_ids == [1] and result.log_probs == [-1.0]


@pytest.mark.asyncio
async def test_observation_preserves_routing_and_marks_missing_abort_time(monkeypatch):
    import numpy as np

    first = segment([1], "abort", "aborted")
    first.routed_experts = np.array([[[10]], [[11]]], dtype=np.uint8)
    first.extra_fields["engine_prefill"] = {"available": True, "seconds": 1.0}
    empty = segment([], None, "aborted")
    final = segment([2], "stop", "completed", "end")
    final.routed_experts = np.array([[[20]], [[21]], [[22]]], dtype=np.uint8)
    final.extra_fields["engine_prefill"] = {"available": True, "seconds": 2.0}
    _install_segments(monkeypatch, [first, empty, final])
    output = await _client().generate(request_id="r", prompt_ids=[10], sampling_params={"max_tokens": 8})
    assert output.token_ids == [1, 2] and output.log_probs == [-1.0, -1.0]
    assert output.routed_experts[:, 0, 0].tolist() == [10, 11, 22]
    summary = output.extra_fields["partial_rollout"]
    assert summary["empty_abort_count"] == 1
    assert summary["resume_count"] == 2
    assert summary["resume_prefill_available"] is False
    assert summary["resume_prefill_observed_seconds"] == 2.0
    assert summary["attempts"][1]["prefill"] == {"available": False, "seconds": None}
