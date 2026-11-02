# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl.utils.reward_score.math_dapo import compute_score, is_correct_minerva


class TestMarkdownEmphasisMinerva:
    """Markdown bold/italic around the Answer: label or the answer value
    must not break Minerva extraction (issue #8449)."""

    def test_bold_label(self):
        result = compute_score("Let me solve it.\n\n**Answer:** 42", "42")
        assert result["acc"] is True

    def test_bold_value(self):
        result = compute_score("Let me solve it.\n\nAnswer: **42**", "42")
        assert result["acc"] is True

    def test_bold_label_newline_answer(self):
        result = compute_score("Let me solve it.\n\n**Answer:**\n\n42", "42")
        assert result["acc"] is True

    def test_italic_label(self):
        result = compute_score("Let me solve it.\n\n*Answer:* 42", "42")
        assert result["acc"] is True

    def test_bold_answer_colon_outside(self):
        result = compute_score("Let me solve it.\n\n**Answer**: 42", "42")
        assert result["acc"] is True

    def test_italic_answer_colon_outside(self):
        result = compute_score("Let me solve it.\n\n*Answer*: 42", "42")
        assert result["acc"] is True

    def test_plain_still_works(self):
        result = compute_score("Let me solve it.\n\nAnswer: 42", "42")
        assert result["acc"] is True

    def test_latex_subscript_preserved(self):
        correct, pred = is_correct_minerva("Answer: x_1", "x_1")
        assert correct is True
        assert "x_1" in pred or pred == "x1"

    def test_math_asterisk_preserved(self):
        correct, pred = is_correct_minerva("Answer: x^*", "x^*")
        assert correct is True

    def test_bold_only_answer_falls_through(self):
        """If the extracted answer is only markdown markers, fall through
        to the boxed fallback."""
        result = compute_score("Thinking...\n\n**Answer:**\n\n\\boxed{42}", "42")
        assert result["acc"] is True
