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

try:
    from math_verify.errors import TimeoutException
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

_GOLD_TARGETS = (LatexExtractionConfig(),)
_PRED_TARGETS = (ExprExtractionConfig(), LatexExtractionConfig())


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        # Use parsing_timeout=None and timeout_seconds=None to disable
        # signal.alarm() which crashes in non-main threads (Ray workers).
        extracted_gold = parse(ground_truth_boxed, _GOLD_TARGETS, parsing_timeout=None)
        extracted_pred = parse(model_output, _PRED_TARGETS, parsing_timeout=None)
        if extracted_gold and extracted_pred:
            ret_score = max(
                1.0 if any(verify(g, p, timeout_seconds=None) for g in extracted_gold) else 0.0 for p in extracted_pred
            )
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass

    return ret_score
