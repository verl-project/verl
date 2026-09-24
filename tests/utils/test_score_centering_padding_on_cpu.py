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
import torch
from tensordict import TensorDict

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.trainer.ppo.score_centering import pad_rollout_topk
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def test_rollout_topk_survives_no_padding_roundtrip_aligned_with_responses():
    k, prompt_width, response_width = 2, 6, 5
    prompt_lengths, response_lengths = [4, 6], [3, 5]
    ids_rows, log_rows, attention, response_mask = [], [], [], []
    for b, (p_len, r_len) in enumerate(zip(prompt_lengths, response_lengths, strict=False)):
        heads = [[100 * b + r, 100 * b + r + 50] for r in range(r_len)]
        log_head = [[-0.5 - r, -1.0] for r in range(r_len)]
        ids, log_probs = pad_rollout_topk(
            heads,
            log_head,
            k=k,
            prompt_width=prompt_width,
            response_width=response_width,
            response_length=r_len,
        )
        ids_rows.append(ids)
        log_rows.append(log_probs)
        mask = torch.zeros(prompt_width + response_width, dtype=torch.int64)
        mask[prompt_width - p_len : prompt_width + r_len] = 1
        attention.append(mask)
        rmask = torch.zeros(response_width, dtype=torch.int64)
        rmask[:r_len] = 1
        response_mask.append(rmask)
    total = prompt_width + response_width
    batch = TensorDict(
        {
            "input_ids": torch.randint(0, 50, (2, total)),
            "attention_mask": torch.stack(attention),
            "response_mask": torch.stack(response_mask),
            "position_ids": torch.arange(total).expand(2, total).clone(),
            "prompts": torch.randint(0, 50, (2, prompt_width)),
            "responses": torch.randint(0, 50, (2, response_width)),
            "rollout_topk_ids": torch.cat(ids_rows),
            "rollout_topk_log_probs": torch.cat(log_rows),
        },
        batch_size=[2],
    )
    data = left_right_2_no_padding(batch)
    assert data["rollout_topk_ids"].is_nested and data["rollout_topk_log_probs"].is_nested
    padded_ids = no_padding_2_padding(data["rollout_topk_ids"], data)
    padded_log_probs = no_padding_2_padding(data["rollout_topk_log_probs"], data)
    assert padded_ids.shape == padded_log_probs.shape == (2, response_width, k)
    for b, r_len in enumerate(response_lengths):
        assert padded_ids[b, :r_len, 0].tolist() == [100 * b + r for r in range(r_len)]
        torch.testing.assert_close(padded_log_probs[b, :r_len, 0], torch.tensor([-0.5 - r for r in range(r_len)]))


def test_padding_template_uses_dummy_heads():
    k = 3
    sample = TensorDict(
        {
            "prompts": torch.zeros(4, dtype=torch.int64),
            "responses": torch.zeros(2, dtype=torch.int64),
            "input_ids": torch.zeros(6, dtype=torch.int64),
            "attention_mask": torch.ones(6, dtype=torch.int64),
            "response_mask": torch.ones(2, dtype=torch.int64),
            "position_ids": torch.arange(6),
            "rollout_topk_ids": torch.zeros(6, k, dtype=torch.int32),
            "rollout_topk_log_probs": torch.zeros(6, k),
        },
        batch_size=[],
    )
    template, _ = construct_minimal_padding_template(sample, {}, eos_token_id=0)
    seq_len = template["input_ids"].shape[0]
    assert template["rollout_topk_ids"].shape == (seq_len, k)
    torch.testing.assert_close(template["rollout_topk_log_probs"].exp().sum(-1), torch.ones(seq_len))
