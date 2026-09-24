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
from dataclasses import dataclass

import pytest

from verl.workers.rollout.utils import extract_response_topk_logprobs


@dataclass
class _Logprob:
    logprob: float
    rank: int


def test_extract_orders_head_by_rank_and_drops_out_of_head_sampled_token():
    step_in_head = {7: _Logprob(-0.1, 1), 3: _Logprob(-1.5, 2), 9: _Logprob(-2.0, 3)}
    step_outside = {3: _Logprob(-0.2, 1), 7: _Logprob(-1.0, 2), 9: _Logprob(-1.9, 3), 42: _Logprob(-8.0, 17)}
    ids, log_probs = extract_response_topk_logprobs([step_in_head, step_outside], k=3)
    assert ids == [[7, 3, 9], [3, 7, 9]]
    assert log_probs == [[-0.1, -1.5, -2.0], [-0.2, -1.0, -1.9]]


def test_extract_rejects_unexpected_head_size():
    with pytest.raises(AssertionError):
        extract_response_topk_logprobs([{1: _Logprob(-0.1, 1)}], k=3)
