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
import importlib.util
from pathlib import Path

import pytest

from verl.utils.reward_score import default_compute_score

_PATH = Path(__file__).resolve().parents[3] / "examples" / "grpo_trainer" / "k31_reward.py"
_spec = importlib.util.spec_from_file_location("k31_reward", _PATH)
k31_reward = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(k31_reward)

END = "<|close|>response<|sep|><|close|>message<|sep|>"
THINK = "<|close|>think<|sep|><|open|>response<|sep|>"


def _score(text, gt="540"):
    return k31_reward.compute_score("math_dapo", text, gt)


class TestTrailingEndMarkers:
    @pytest.mark.parametrize(
        "text,gt,acc",
        [
            ("work" + THINK + "So the total is 540.\n\nAnswer: 540" + END, "540", True),
            ("work" + THINK + "Answer: $540$" + END, "540", True),
            ("work" + THINK + "Answer: 540 <|close|> response <|sep|> <|close|> message <|sep|>\n", "540", True),
            ("work" + THINK + "Answer: 541" + END, "540", False),
            ("work" + THINK + r"so the answer is \boxed{45}" + END, "45", True),
            ("Answer: 540", "540", True),
            ("still thinking about 540 and", "540", False),
        ],
        ids=["plain", "dollar", "spaced-markers", "wrong", "boxed-fallback", "no-markers", "truncated"],
    )
    def test_grade(self, text, gt, acc):
        assert _score(text, gt)["acc"] is acc

    def test_plain_answer_was_graded_wrong_before_the_strip(self):
        text = "work" + THINK + "Answer: 540" + END
        assert default_compute_score("math_dapo", text, "540")["acc"] is False
        assert default_compute_score("math_dapo", "work" + THINK + "Answer: $540$" + END, "540")["acc"] is True

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("a" + THINK + "Answer: 7" + END, "a" + THINK + "Answer: 7"),
            ("a" + THINK + "Answer: 7", "a" + THINK + "Answer: 7"),
            ("a<|close|>think<|sep|>", "a<|close|>think<|sep|>"),
            ("Answer: 7" + END + " tail", "Answer: 7" + END + " tail"),
        ],
        ids=["strips-end-run", "nothing-to-strip", "keeps-think-close", "only-anchored-at-end"],
    )
    def test_strip_is_anchored(self, text, expected):
        assert k31_reward.strip_trailing_end_markers(text) == expected
