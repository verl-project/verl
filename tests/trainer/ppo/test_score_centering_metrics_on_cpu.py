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

"""Token-weighted SC diagnostics across uneven microbatches and data-parallel ranks."""

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding


def _microbatch_metrics(response_length, aligned, dp_size, global_tokens, global_batch_size):
    prompt_length = 2
    sequence_length = prompt_length + response_length
    data = TensorDict(
        {
            "prompts": torch.zeros(1, prompt_length, dtype=torch.long),
            "responses": torch.zeros(1, response_length, dtype=torch.long),
            "input_ids": torch.zeros(1, sequence_length, dtype=torch.long),
            "attention_mask": torch.ones(1, sequence_length, dtype=torch.long),
            "position_ids": torch.arange(sequence_length).unsqueeze(0),
            "response_mask": torch.ones(1, response_length, dtype=torch.long),
            "old_log_probs": torch.zeros(1, response_length),
            "advantages": torch.zeros(1, response_length),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, dp_size=dp_size, batch_num_tokens=global_tokens, global_batch_size=global_batch_size)
    data = left_right_2_no_padding(data)

    def packed_response_metric(value):
        # Non-response prediction rows have deliberately invalid diagnostic values;
        # ppo_loss must exclude them before aggregating the global response-token fraction.
        row = torch.full((sequence_length,), 999.0)
        row[prompt_length - 1 : prompt_length - 1 + response_length] = value
        return torch.nested.as_nested_tensor([row], layout=torch.jagged)

    zero = torch.nested.as_nested_tensor([torch.zeros(sequence_length)], layout=torch.jagged)
    model_output = {
        "log_probs": zero,
        "sc_correction": zero,
        "sc_support_aligned": packed_response_metric(float(aligned)),
        "sc_support_unknown": packed_response_metric(float(not aligned)),
        "sc_sampler_tail_mass": packed_response_metric(0.0 if aligned else 0.25),
    }
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(
            loss_mode="bypass_mode", rollout_correction=RolloutCorrectionConfig.bypass_pg_sc()
        ),
        loss_agg_mode="token-mean",
    )
    _, metrics = ppo_loss(config, model_output, data)
    return metrics


@pytest.mark.parametrize(
    "rank_microbatches",
    [
        (((1, True), (100, False)),),
        (((1, True),), ((100, False),)),
        (((1, True), (49, False)), ((1, False), (50, False))),
    ],
    ids=["uneven-microbatches", "uneven-dp-ranks", "uneven-microbatches-and-dp"],
)
def test_score_centering_diagnostics_use_the_global_response_token_denominator(rank_microbatches):
    global_tokens = sum(length for rank in rank_microbatches for length, _ in rank)
    global_batch_size = sum(len(rank) for rank in rank_microbatches)
    dp_size = len(rank_microbatches)
    per_rank = []
    for microbatches in rank_microbatches:
        rank_metrics = {}
        for length, aligned in microbatches:
            metrics = _microbatch_metrics(length, aligned, dp_size, global_tokens, global_batch_size)
            for key in ("actor/sc_support_aligned", "actor/sc_support_unknown", "actor/sc_sampler_tail_mass"):
                assert metrics[key].aggregation is AggregationType.SUM
                rank_metrics.setdefault(key, metrics[key].init_list()).extend(metrics[key])
        per_rank.append(rank_metrics)
    # This is the same rank reduction used by TrainingWorker.train_mini_batch.
    combined = {key: Metric.aggregate_dp([rank[key] for rank in per_rank]) for key in per_rank[0]}
    assert combined["actor/sc_support_aligned"] == pytest.approx(1 / 101)
    assert combined["actor/sc_support_unknown"] == pytest.approx(100 / 101)
    assert combined["actor/sc_sampler_tail_mass"] == pytest.approx(25 / 101)
