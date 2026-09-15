"""Pre-filter telemetry must count refills, not only selected training samples."""

import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import accumulate_rollout_timing_metrics, accumulate_rollout_workload_metrics


def batch(lengths):
    mask = torch.arange(8)[None, :] < torch.tensor(lengths, dtype=torch.long)[:, None]
    return DataProto.from_dict(tensors={"response_mask": mask})


@pytest.mark.parametrize("lengths", [[1, 8, 2], [0, 0], [8], []])
def test_single_batch(lengths):
    metrics = accumulate_rollout_workload_metrics(batch(lengths), {})
    assert metrics["rollout/pre_filter/sequences"] == len(lengths)
    assert metrics["rollout/pre_filter/response_tokens"] == sum(lengths)
    assert metrics["rollout/pre_filter/response_length_max"] == max(lengths, default=0)
    assert metrics["rollout/pre_filter/response_length_mean"] == (sum(lengths) / len(lengths) if lengths else 0)


def test_refill_counts_and_weighted_mean_without_mutating_previous():
    first = accumulate_rollout_workload_metrics(batch([1, 8, 2]), {})
    saved = dict(first)
    result = accumulate_rollout_workload_metrics(batch([3]), first)
    assert first == saved
    assert result["rollout/pre_filter/sequences"] == 4
    assert result["rollout/pre_filter/response_tokens"] == 14
    assert result["rollout/pre_filter/response_length_mean"] == 3.5
    assert result["rollout/pre_filter/response_length_max"] == 8
    # Policy-update reset must not inherit discarded/refill tokens.
    reset = accumulate_rollout_workload_metrics(batch([3]), {})
    assert reset["rollout/pre_filter/response_tokens"] == 3


def test_observation_and_padding_tokens_are_not_counted():
    data = DataProto.from_dict(tensors={"response_mask": torch.tensor([[1, 0, 1, 0]])})
    assert accumulate_rollout_workload_metrics(data, {})["rollout/pre_filter/response_tokens"] == 2


def test_refill_timing_uses_weighted_means_and_coherent_slowest_sample():
    first = {
        "gen": 40.0,
        "agent_loop/generate_sequences/min": 1.0,
        "agent_loop/generate_sequences/max": 30.0,
        "agent_loop/generate_sequences/mean": 10.0,
        "agent_loop/slowest/generate_sequences": 30.0,
        "agent_loop/slowest/tool_calls": 0.0,
        "agent_loop/slowest/compute_score": 0.0,
        "agent_loop/slowest/response_length": 300.0,
    }
    second = {
        "agent_loop/generate_sequences/min": 2.0,
        "agent_loop/generate_sequences/max": 20.0,
        "agent_loop/generate_sequences/mean": 5.0,
        "agent_loop/slowest/generate_sequences": 20.0,
        "agent_loop/slowest/tool_calls": 11.0,
        "agent_loop/slowest/compute_score": 0.0,
        "agent_loop/slowest/response_length": 200.0,
    }
    result = accumulate_rollout_timing_metrics(first, second, previous_sequences=3, sequences=1)
    assert result["gen"] == 40.0  # marked_timer owns stage wall-clock accumulation
    assert result["agent_loop/generate_sequences/min"] == 1.0
    assert result["agent_loop/generate_sequences/max"] == 30.0
    assert result["agent_loop/generate_sequences/mean"] == 8.75
    assert result["agent_loop/slowest/generate_sequences"] == 20.0
    assert result["agent_loop/slowest/response_length"] == 200.0
    assert first["agent_loop/generate_sequences/mean"] == 10.0
    # A shorter later refill must not overwrite the earlier slowest trajectory.
    shorter = accumulate_rollout_timing_metrics(second, first, previous_sequences=1, sequences=3)
    assert shorter["agent_loop/slowest/response_length"] == 200.0
    assert shorter["agent_loop/generate_sequences/mean"] == 8.75


def test_refill_timing_empty_and_update_reset():
    values = {"agent_loop/generate_sequences/mean": 4.0}
    assert accumulate_rollout_timing_metrics({}, values, previous_sequences=0, sequences=2) == values
    assert accumulate_rollout_timing_metrics(values, {}, previous_sequences=2, sequences=0) == values
    with pytest.raises(ValueError, match="nonnegative"):
        accumulate_rollout_timing_metrics({}, {}, previous_sequences=-1, sequences=1)
