# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
from types import SimpleNamespace

import pytest

from verl.utils.metric.partial_rollout import partial_rollout_metrics, read_partial_rollout_fields
from verl.workers.rollout.partial_metrics import engine_prefill_timing, summarize_partial_attempts


@pytest.mark.parametrize("start,end", [(None, None), (0, 0), (5, 0), (5, 4), (float("nan"), 6)])
def test_missing_or_incomplete_prefill_is_unavailable(start, end):
    assert engine_prefill_timing(SimpleNamespace(scheduled_ts=start, first_token_ts=end)) == {
        "available": False,
        "seconds": None,
    }


def test_prefill_interval_excludes_arrival_and_decode():
    metrics = SimpleNamespace(arrival_time=1, scheduled_ts=5, first_token_ts=7, last_token_ts=100)
    assert engine_prefill_timing(metrics) == {"available": True, "seconds": 2}


def test_partial_aggregation_preserves_unavailable_and_excludes_padding():
    def attempt(retained, tokens, aborted, timing):
        return {"retained_tokens": retained, "new_tokens": tokens, "aborted": aborted, "prefill": timing}

    available = {"available": True, "seconds": 2.5}
    missing = {"available": False, "seconds": None}
    resumed = summarize_partial_attempts(
        [attempt(0, 2, True, available), attempt(2, 0, True, missing), attempt(2, 3, False, available)]
    )
    plain = summarize_partial_attempts([attempt(0, 5, False, available)])
    fields = [{"partial_rollout": resumed}, {"partial_rollout": plain}, {}, {"partial_rollout": resumed}]
    result = partial_rollout_metrics(fields, [True, True, True, False])
    p = "training/partial_rollout/"
    assert result[p + "response_coverage"] == pytest.approx(2 / 3)
    assert result[p + "abort_count"] == result[p + "resume_count"] == 2
    assert result[p + "empty_abort_count"] == 1
    assert result[p + "resumed_response_fraction"] == 0.5
    assert result[p + "retained_prefix_response_fraction"] == 0.5
    assert result[p + "resume_prefill_coverage"] == 0.5
    assert result[p + "resume_prefill_available"] == 0
    assert result[p + "resume_prefill_observed_seconds"] == 2.5
    assert resumed["resume_prefill_available"] is False
    complete = summarize_partial_attempts([attempt(0, 2, True, available), attempt(2, 3, False, available)])
    assert complete["resume_prefill_available"] is True
    assert complete["resume_prefill_observed_seconds"] == 2.5


@pytest.mark.parametrize("partition", ["train", "training-custom"])
def test_optional_observations_preserve_partition_and_missing_row_coverage(partition):
    import numpy as np

    record = {"partial_rollout": summarize_partial_attempts([])}
    calls = []

    def read(*, keys, partition_id, select_fields):
        assert partition_id == partition
        assert select_fields == ["extra_fields"]
        calls.append(keys)
        if len(keys) > 1 or keys == ["old"]:
            raise ValueError("Some fields are not ready in all the requested keys!")
        return {"extra_fields": np.array([record], dtype=object)}

    observations = read_partial_rollout_fields(["new", "old"], partition, read)
    assert calls == [["new", "old"], ["new"], ["old"]]
    metrics = partial_rollout_metrics(observations, [True, True])
    assert metrics["training/partial_rollout/response_coverage"] == 0.5
    assert metrics["training/partial_rollout/abort_count"] == 0


def test_optional_observations_propagate_unrelated_queue_error():
    def read(**kwargs):
        raise ValueError("partition does not exist")

    with pytest.raises(ValueError, match="partition does not exist"):
        read_partial_rollout_fields(["new"], "training-custom", read)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_client_retains_attempt_history_with_output_for_checkpointing(monkeypatch, enabled):
    from verl.workers.rollout import llm_server
    from verl.workers.rollout.replica import TokenOutput

    outputs = iter(
        [
            TokenOutput(token_ids=[3, 4], stop_reason="aborted", extra_fields={"global_steps": 7}),
            TokenOutput(token_ids=[], stop_reason="abort", extra_fields={"global_steps": 8}),
            TokenOutput(token_ids=[5], stop_reason="stop", extra_fields={"global_steps": 9}),
        ]
    )

    async def generate(self, **kwargs):
        return next(outputs)

    async def no_wait(*args):
        pass

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", generate)
    monkeypatch.setattr(llm_server.asyncio, "sleep", no_wait)
    config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(collect_partial_rollout_metrics=enabled, response_length=5)
        )
    )
    output = await llm_server.FullyAsyncLLMServerClient(config=config).generate(
        "request", prompt_ids=[1, 2], sampling_params={"max_tokens": 5}
    )
    assert output.token_ids == [3, 4, 5]
    if enabled:
        history = output.model_dump()["extra_fields"]["partial_attempts"]
        assert [attempt["version"] for attempt in history] == [7, 8, 9]
        assert [attempt["retained_tokens"] for attempt in history] == [0, 2, 2]
        assert [attempt["new_tokens"] for attempt in history] == [2, 0, 1]
        assert output.extra_fields["partial_rollout"]["abort_count"] == 2
        assert output.extra_fields["partial_rollout"]["empty_abort_count"] == 1
        assert len(output.extra_fields["termination"]["attempts"]) == len(history)
    else:
        assert "partial_attempts" not in output.extra_fields
        assert "partial_rollout" not in output.extra_fields
