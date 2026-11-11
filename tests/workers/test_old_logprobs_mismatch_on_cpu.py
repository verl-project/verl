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

"""Per-token mismatch count between the update forward log-probs and old_log_probs."""

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding

PROMPT_LENGTH = 2


def _metrics(old_log_probs, log_probs, dp_size=1, batch_num_tokens=None, global_batch_size=None):
    response_length = old_log_probs.numel()
    sequence_length = PROMPT_LENGTH + response_length
    data = TensorDict(
        {
            "prompts": torch.zeros(1, PROMPT_LENGTH, dtype=torch.long),
            "responses": torch.zeros(1, response_length, dtype=torch.long),
            "input_ids": torch.zeros(1, sequence_length, dtype=torch.long),
            "attention_mask": torch.ones(1, sequence_length, dtype=torch.long),
            "position_ids": torch.arange(sequence_length).unsqueeze(0),
            "response_mask": torch.ones(1, response_length, dtype=torch.long),
            "old_log_probs": old_log_probs.unsqueeze(0),
            "advantages": torch.zeros(1, response_length),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size)
    data = left_right_2_no_padding(data)
    row = torch.zeros(sequence_length)
    row[PROMPT_LENGTH - 1 : PROMPT_LENGTH - 1 + response_length] = log_probs
    model_output = {"log_probs": torch.nested.as_nested_tensor([row], layout=torch.jagged)}
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, loss_agg_mode="token-mean")
    _, metrics = ppo_loss(config, model_output, data)
    return metrics


def test_equal_log_probs_count_zero():
    old = torch.tensor([-1.0, -2.0, -3.0])
    metrics = _metrics(old, old.clone())
    assert metrics["actor/old_logprobs_mismatch_count"].aggregation is AggregationType.SUM
    assert metrics["actor/old_logprobs_mismatch_count"].aggregate() == 0


def test_offsetting_differences_cancel_in_ppo_kl_but_are_counted():
    old = torch.tensor([-1.0, -2.0, -3.0])
    new = torch.tensor([-1.5, -1.5, -3.0])
    metrics = _metrics(old, new)
    assert metrics["actor/ppo_kl"].aggregate() == 0.0
    assert metrics["actor/old_logprobs_mismatch_count"].aggregate() == 2


def test_one_ulp_difference_is_counted():
    old = torch.tensor([-1.0, -2.0], dtype=torch.bfloat16)
    new = old.clone()
    new[1] = torch.nextafter(new[1], torch.tensor(0.0, dtype=torch.bfloat16))
    metrics = _metrics(old.float(), new.float())
    assert metrics["actor/old_logprobs_mismatch_count"].aggregate() == 1


def test_dp_reduction_recovers_the_global_count():
    old = torch.tensor([-1.0, -2.0, -3.0])
    ranks = [
        _metrics(old, old + torch.tensor([1e-3, 0.0, 0.0]), dp_size=2, batch_num_tokens=6, global_batch_size=2)
        for _ in range(2)
    ]
    # This is the same rank reduction used by TrainingWorker.train_mini_batch.
    combined = Metric.aggregate_dp([rank["actor/old_logprobs_mismatch_count"] for rank in ranks])
    assert combined == 2
