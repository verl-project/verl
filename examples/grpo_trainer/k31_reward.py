# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Rule reward for the K31 recipe: VERL's default scorer on the response without its end markers.

K31's chat markers (``<|open|>``, ``<|close|>``, ``<|sep|>``) are ordinary vocabulary pieces, not
special tokens, so ``decode(skip_special_tokens=True)`` keeps them and a finished response ends with
``Answer: 540<|close|>response<|sep|><|close|>message<|sep|>``. The DAPO rule takes the whole
``Answer:`` line, so a plain ``540`` is graded wrong while ``$540$`` is graded right (the normalizer
keeps only the text between the dollar signs): the reward then trains the answer format, not the
answer.

Only the run of response/message end markers at the very end of the text is removed. The same
markers close the thinking channel earlier in the text; cutting at their first occurrence would
grade the thinking part instead of the answer.

Usage: ``reward.custom_reward_function.path=<this file> reward.custom_reward_function.name=compute_score``.
"""

import re

from verl.utils.reward_score import default_compute_score

_TRAILING_END_MARKERS = re.compile(r"(?:\s*<\|close\|>\s*(?:response|message)\s*<\|sep\|>)+\s*$")


def strip_trailing_end_markers(text: str) -> str:
    """Remove the response/message end markers that terminate a finished K31 response."""
    return _TRAILING_END_MARKERS.sub("", text)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return default_compute_score(
        data_source, strip_trailing_end_markers(solution_str), ground_truth, extra_info, **kwargs
    )
