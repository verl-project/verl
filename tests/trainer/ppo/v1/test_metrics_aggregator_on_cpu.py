# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""CPU-only unit tests for ``verl.trainer.ppo.v1.utils.MetricsAggregator``.

``separate_async`` runs ``parameter_sync_step`` local updates inside a single
``global_step``; ``MetricsAggregator`` reduces the per-iteration training metrics
into one value per key before they are logged once for the step. These tests pin
down the per-key aggregation rules (weighted mean/sum/max/min/last/time_sum).
"""

import math

import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY
from verl.trainer.ppo.v1.utils import MetricsAggregator
from verl.utils.debug.metrics import calculate_debug_metrics

_ROLLOUT_DEBUG_AVERAGES = (
    "training/rollout_probs_diff_mean",
    "training/rollout_probs_diff_std",
    "training/rollout_actor_probs_pearson_corr",
    "training/rollout_logprobs_diff_mean",
)
_ROLLOUT_DEBUG_MAXIMA = (
    "training/rollout_probs_diff_max",
    "training/rollout_logprobs_diff_max",
)
_ROLLOUT_DEBUG_VALID = "training/rollout_probs_diff_valid"


def test_empty_aggregator_returns_empty():
    agg = MetricsAggregator()
    assert agg.get_aggregated_metrics() == {}


def test_mean_metrics_are_weighted_by_sample_count():
    agg = MetricsAggregator()
    agg.add_step_metrics({"actor/pg_loss/mean": 1.0}, sample_count=1)
    agg.add_step_metrics({"actor/pg_loss/mean": 3.0}, sample_count=3)
    out = agg.get_aggregated_metrics()
    assert out["actor/pg_loss/mean"] == pytest.approx(2.5)


def test_default_metric_is_weighted_by_sample_count():
    # No aggregation keyword in the name -> defaults to weighted average.
    agg = MetricsAggregator()
    agg.add_step_metrics({"actor/kl": 2.0}, sample_count=1)
    agg.add_step_metrics({"actor/kl": 4.0}, sample_count=3)
    assert agg.get_aggregated_metrics()["actor/kl"] == pytest.approx(3.5)


def test_max_and_min_are_reduced():
    agg = MetricsAggregator()
    for v in (1.0, 5.0, 3.0):
        agg.add_step_metrics({"grad/max": v, "grad/min": v})
    out = agg.get_aggregated_metrics()
    assert out["grad/max"] == 5.0
    assert out["grad/min"] == 1.0


def test_timing_and_sum_metrics_accumulate():
    agg = MetricsAggregator()
    agg.add_step_metrics({"timing_s/update_actor": 1.5, "some/total_tokens": 10})
    agg.add_step_metrics({"timing_s/update_actor": 2.5, "some/total_tokens": 20})
    out = agg.get_aggregated_metrics()
    assert out["timing_s/update_actor"] == pytest.approx(4.0)
    assert out["some/total_tokens"] == pytest.approx(30.0)


def test_evicted_samples_are_summed_across_iterations():
    # Per-iteration off-policy eviction counts must accumulate, not average.
    agg = MetricsAggregator()
    agg.add_step_metrics({"training/off_policy/evicted_samples": 2})
    agg.add_step_metrics({"training/off_policy/evicted_samples": 3})
    assert agg.get_aggregated_metrics()["training/off_policy/evicted_samples"] == pytest.approx(5.0)


def test_evicted_samples_staleness_mean_uses_evicted_count_weight():
    # Evicted-sample staleness is averaged over evicted samples, not over kept batch size.
    agg = MetricsAggregator()
    agg.add_step_metrics(
        {
            "training/off_policy/evicted_samples": 1,
            "training/off_policy/evicted_samples_staleness/mean": 10.0,
        },
        sample_count=100,
    )
    agg.add_step_metrics(
        {
            "training/off_policy/evicted_samples": 3,
            "training/off_policy/evicted_samples_staleness/mean": 2.0,
        },
        sample_count=1,
    )
    assert agg.get_aggregated_metrics()["training/off_policy/evicted_samples_staleness/mean"] == pytest.approx(4.0)


def test_last_metric_keeps_final_value():
    agg = MetricsAggregator()
    agg.add_step_metrics({"training/global_step": 7})
    agg.add_step_metrics({"training/global_step": 8})
    assert agg.get_aggregated_metrics()["training/global_step"] == 8


@pytest.mark.parametrize("flags", [(1, 0), (0, 1), (0, 0)])
def test_rollout_probs_diff_valid_reports_any_available_step(flags):
    agg = MetricsAggregator()
    for flag in flags:
        agg.add_step_metrics({_ROLLOUT_DEBUG_VALID: flag})
    assert agg.get_aggregated_metrics()[_ROLLOUT_DEBUG_VALID] == max(flags)


@pytest.mark.parametrize("metric", _ROLLOUT_DEBUG_AVERAGES + _ROLLOUT_DEBUG_MAXIMA)
@pytest.mark.parametrize("empty_position", [0, 1, 2])
@pytest.mark.parametrize("empty_weight", [0, 100])
def test_rollout_debug_metrics_exclude_unavailable_steps(metric, empty_position, empty_weight):
    steps = [(1, 2.0, 1), (1, 6.0, 3)]
    steps.insert(empty_position, (0, float("nan"), empty_weight))
    agg = MetricsAggregator()
    for valid, value, weight in steps:
        agg.add_step_metrics({metric: value, _ROLLOUT_DEBUG_VALID: valid}, sample_count=weight)
    out = agg.get_aggregated_metrics()
    assert out[metric] == pytest.approx(6.0 if metric in _ROLLOUT_DEBUG_MAXIMA else 5.0)
    assert out[_ROLLOUT_DEBUG_VALID] == 1


@pytest.mark.parametrize("metric", _ROLLOUT_DEBUG_AVERAGES + _ROLLOUT_DEBUG_MAXIMA)
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("nonfinite_first", [False, True])
def test_rollout_debug_metrics_preserve_available_nonfinite_values(metric, nonfinite, nonfinite_first):
    steps = [(1, 2.0), (1, nonfinite)]
    if nonfinite_first:
        steps.reverse()
    # A final empty step must neither hide the failure nor clear aggregate validity.
    steps.append((0, float("nan")))
    agg = MetricsAggregator()
    for valid, value in steps:
        agg.add_step_metrics({metric: value, _ROLLOUT_DEBUG_VALID: valid}, sample_count=1)
    out = agg.get_aggregated_metrics()
    assert out[_ROLLOUT_DEBUG_VALID] == 1
    if math.isnan(nonfinite):
        assert math.isnan(out[metric])
    else:
        assert out[metric] == nonfinite


@pytest.mark.parametrize("sample_count", [0, 2])
def test_rollout_debug_metrics_all_unavailable(sample_count):
    metrics = dict.fromkeys(_ROLLOUT_DEBUG_AVERAGES + _ROLLOUT_DEBUG_MAXIMA, float("nan"))
    metrics.update({_ROLLOUT_DEBUG_VALID: 0, "training/rollout_logprobs_mismatch_count": 0})
    agg = MetricsAggregator()
    agg.add_step_metrics(metrics, sample_count=sample_count)
    agg.add_step_metrics(metrics, sample_count=sample_count)
    out = agg.get_aggregated_metrics()
    assert out.keys() == metrics.keys()
    assert out[_ROLLOUT_DEBUG_VALID] == 0
    assert out["training/rollout_logprobs_mismatch_count"] == 0
    for key in _ROLLOUT_DEBUG_AVERAGES + _ROLLOUT_DEBUG_MAXIMA:
        assert math.isnan(out[key])
    agg.reset()
    assert agg.get_aggregated_metrics() == {}
    agg.add_step_metrics({"actor/kl": 2.0})
    assert agg.get_aggregated_metrics() == {"actor/kl": 2.0}


@pytest.mark.parametrize("metric", _ROLLOUT_DEBUG_AVERAGES)
def test_rollout_debug_metrics_preserve_zero_total_weight_fallback(metric):
    agg = MetricsAggregator()
    agg.add_step_metrics({metric: 2.0, _ROLLOUT_DEBUG_VALID: 1}, sample_count=0)
    agg.add_step_metrics({metric: float("nan"), _ROLLOUT_DEBUG_VALID: 0}, sample_count=10)
    agg.add_step_metrics({metric: 6.0, _ROLLOUT_DEBUG_VALID: 1}, sample_count=0)
    assert agg.get_aggregated_metrics()[metric] == pytest.approx(4.0)


def test_rollout_debug_metrics_missing_guard_and_missing_metrics():
    agg = MetricsAggregator()
    metric = "training/rollout_probs_diff_mean"
    agg.add_step_metrics({metric: 2.0}, sample_count=1)
    agg.add_step_metrics({}, sample_count=100)
    agg.add_step_metrics({metric: 6.0}, sample_count=3)
    assert agg.get_aggregated_metrics() == {metric: 5.0}


def test_rollout_debug_metrics_accept_scalar_tensors():
    agg = MetricsAggregator()
    metric = "training/rollout_probs_diff_mean"
    agg.add_step_metrics({metric: torch.tensor(float("nan")), _ROLLOUT_DEBUG_VALID: torch.tensor(0)})
    agg.add_step_metrics({metric: torch.tensor(2.0), _ROLLOUT_DEBUG_VALID: torch.tensor(1)}, sample_count=2)
    assert agg.get_aggregated_metrics() == {metric: 2.0, _ROLLOUT_DEBUG_VALID: 1.0}


@pytest.mark.parametrize("nonfinite", [False, True])
def test_calculated_rollout_debug_metrics_aggregate_across_empty_responses(nonfinite):
    agg = MetricsAggregator()
    valid_metrics = []
    for available, weight in [(True, 1), (False, 10), (True, 3), (False, 10)]:
        actor = torch.tensor([[-1.5, -2.5]])
        if available and nonfinite:
            actor[0, 0] = float("nan")
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor([[-1.0, -2.0]]),
                "old_log_probs": actor,
                "response_mask": torch.full((1, 2), int(available)),
                "responses": torch.zeros((1, 2)),
            }
        )
        metrics = calculate_debug_metrics(data)
        if available:
            valid_metrics.append(metrics)
        agg.add_step_metrics(metrics, sample_count=weight)
    out = agg.get_aggregated_metrics()
    assert out[_ROLLOUT_DEBUG_VALID] == 1
    assert out["training/rollout_logprobs_mismatch_count"] == 4
    for key in _ROLLOUT_DEBUG_AVERAGES + _ROLLOUT_DEBUG_MAXIMA:
        if nonfinite:
            assert math.isnan(out[key])
        else:
            # Identical available batches retain their diagnostics, regardless of empty batches' weights.
            assert out[key] == pytest.approx(valid_metrics[0][key])


def test_rollout_logprobs_mismatch_count_sums():
    # a token count: the gate needs the total over the aggregation window, not the mean per iteration.
    agg = MetricsAggregator()
    agg.add_step_metrics({"training/rollout_logprobs_mismatch_count": 0})
    agg.add_step_metrics({"training/rollout_logprobs_mismatch_count": 3})
    assert agg.get_aggregated_metrics()["training/rollout_logprobs_mismatch_count"] == 3


def test_global_seqlen_minmax_diff_is_recomputed_from_aggregated_min_max():
    # minmax_diff must be recomputed as (aggregated max - aggregated min), not reduced by the
    # "max" substring heuristic over the per-iteration diffs.
    agg = MetricsAggregator()
    agg.add_step_metrics({"global_seqlen/min": 10.0, "global_seqlen/max": 20.0, "global_seqlen/minmax_diff": 10.0})
    agg.add_step_metrics({"global_seqlen/min": 0.0, "global_seqlen/max": 15.0, "global_seqlen/minmax_diff": 15.0})
    out = agg.get_aggregated_metrics()
    assert out["global_seqlen/min"] == 0.0
    assert out["global_seqlen/max"] == 20.0
    # recomputed: 20 - 0 = 20 (naive max over per-iter diffs would give 15).
    assert out["global_seqlen/minmax_diff"] == pytest.approx(20.0)


def test_minmax_diff_falls_back_when_min_max_absent():
    # Without global_seqlen/min and /max present, minmax_diff keeps the heuristic reduction (max).
    agg = MetricsAggregator()
    agg.add_step_metrics({"global_seqlen/minmax_diff": 7.0})
    agg.add_step_metrics({"global_seqlen/minmax_diff": 3.0})
    assert agg.get_aggregated_metrics()["global_seqlen/minmax_diff"] == pytest.approx(7.0)


def test_tensor_and_non_scalar_handling():
    agg = MetricsAggregator()
    # 0-d / single-element tensors are recorded; multi-element tensors are ignored.
    agg.add_step_metrics({"actor/loss/mean": torch.tensor(2.0), "actor/vec": torch.tensor([1.0, 2.0])}, sample_count=1)
    agg.add_step_metrics({"actor/loss/mean": torch.tensor(4.0)}, sample_count=3)
    out = agg.get_aggregated_metrics()
    assert out["actor/loss/mean"] == pytest.approx(3.5)
    assert "actor/vec" not in out


def test_bool_values_are_ignored():
    agg = MetricsAggregator()
    agg.add_step_metrics({"flag": True, "actor/loss/mean": 1.0})
    out = agg.get_aggregated_metrics()
    assert "flag" not in out
    assert out["actor/loss/mean"] == pytest.approx(1.0)


def test_reset_clears_state():
    agg = MetricsAggregator()
    agg.add_step_metrics({"actor/loss/mean": 1.0})
    agg.reset()
    assert agg.get_aggregated_metrics() == {}
    assert agg.step_count == 0


def test_missing_key_in_some_iterations_uses_present_values():
    # A metric only reported on some iterations uses the weights from the iterations where it exists.
    agg = MetricsAggregator()
    agg.add_step_metrics({"actor/loss/mean": 2.0}, sample_count=1)
    agg.add_step_metrics({}, sample_count=100)  # e.g. critic_warmup step with no actor update
    agg.add_step_metrics({"actor/loss/mean": 6.0}, sample_count=3)
    out = agg.get_aggregated_metrics()
    assert out["actor/loss/mean"] == pytest.approx(5.0)
    assert not math.isnan(out["actor/loss/mean"])


# --------------------------------------------------------------------------- #
# DAPO group-filtering diagnostics flow through the aggregator (multiple
# mini-batches per global step). These pin down two suspected bugs:
#   Bug #1: the dict-valued reward-count breakdown is dropped, so the wandb table
#           never receives data on the real step() -> aggregator path.
#   Bug #2: filter_groups / rollout_failure eviction *counts* are averaged
#           instead of summed across mini-batches.
# --------------------------------------------------------------------------- #


def test_dapo_filtered_reward_counts_survive_aggregation():
    """The {reward_value: count} breakdown must reach the logger via the aggregator.

    ``ReplayBuffer.sample`` emits ``DAPO_FILTERED_REWARD_COUNTS_KEY`` as a dict; the trainer pops it
    from the aggregated metrics and feeds the wandb table. If the aggregator drops it (only scalars
    kept), the table is never populated in the real training loop.
    """
    agg = MetricsAggregator()
    agg.add_step_metrics({DAPO_FILTERED_REWARD_COUNTS_KEY: {0.0: 3, 1.0: 1}}, sample_count=8)
    agg.add_step_metrics({DAPO_FILTERED_REWARD_COUNTS_KEY: {0.0: 2}}, sample_count=8)

    out = agg.get_aggregated_metrics()
    assert DAPO_FILTERED_REWARD_COUNTS_KEY in out, (
        "dict-valued DAPO breakdown was dropped by the aggregator; wandb table never gets data"
    )
    # Counts for the same step must accumulate additively across mini-batches.
    assert out[DAPO_FILTERED_REWARD_COUNTS_KEY] == {0.0: 5, 1.0: 1}


def test_filter_groups_evicted_samples_are_summed_across_iterations():
    # Per-mini-batch DAPO eviction counts must accumulate, like off_policy/evicted_samples.
    agg = MetricsAggregator()
    agg.add_step_metrics({"training/filter_groups/evicted_samples": 2})
    agg.add_step_metrics({"training/filter_groups/evicted_samples": 3})
    assert agg.get_aggregated_metrics()["training/filter_groups/evicted_samples"] == pytest.approx(5.0)


def test_filter_groups_discarded_surplus_samples_are_summed_across_iterations():
    agg = MetricsAggregator()
    agg.add_step_metrics({"training/filter_groups/discarded_surplus_samples": 4})
    agg.add_step_metrics({"training/filter_groups/discarded_surplus_samples": 1})
    assert agg.get_aggregated_metrics()["training/filter_groups/discarded_surplus_samples"] == pytest.approx(5.0)


def test_rollout_failure_evicted_samples_are_summed_across_iterations():
    agg = MetricsAggregator()
    agg.add_step_metrics({"training/rollout_failure/evicted_samples": 1})
    agg.add_step_metrics({"training/rollout_failure/evicted_samples": 2})
    assert agg.get_aggregated_metrics()["training/rollout_failure/evicted_samples"] == pytest.approx(3.0)
