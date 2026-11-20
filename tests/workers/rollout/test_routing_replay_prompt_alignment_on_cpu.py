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

import pytest

from verl.workers.rollout.utils import routing_replay_prompts_match


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "compact,expanded",
    [
        ([1, 10, 2], [1] + [10] * 4 + [2]),
        ([1, 10, 10, 2], [1] + [10] * 9 + [2]),
        ([1, 10, 10, 2, 2, 10, 3], [1] + [10] * 8 + [2, 2] + [10] * 4 + [3]),
    ],
)
def test_replay_allows_one_compact_image_placeholder_per_image(reverse, compact, expanded):
    actor, rollout = (expanded, compact) if reverse else (compact, expanded)
    assert routing_replay_prompts_match(actor, rollout, image_token_id=10, image_count=compact.count(10))


@pytest.mark.parametrize(
    "actor,rollout,count",
    [
        ([1, 10, 10, 2], [1, 10, 10, 10, 2], 1),
        ([1, 10, 10, 2, 10, 10, 3], [1, 10, 10, 10, 2, 10, 3], 2),
        ([1, 10, 10, 2, 10, 3], [1, 10, 2, 10, 10, 3], 3),
        ([1, 10, 10, 2], [1, 10, 2], 2),
        ([1, 10, 10, 2, 10, 3], [1] + [10] * 4 + [2] + [10] * 8 + [3], 3),
        ([1, 10, 2], [1, 10, 10, 3], 1),
        ([1, 10, 2], [1, 10, 10, 2], 0),
    ],
)
def test_replay_rejects_mismatched_expanded_or_compact_prompts(actor, rollout, count):
    assert not routing_replay_prompts_match(actor, rollout, image_token_id=10, image_count=count)


def test_other_replay_prompts_require_exact_token_equality():
    assert routing_replay_prompts_match([1, 10, 10, 2], [1, 10, 10, 2])
    assert not routing_replay_prompts_match([1, 10, 2], [1, 10, 10, 2])
