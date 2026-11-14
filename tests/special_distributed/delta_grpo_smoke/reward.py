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
"""Synthetic bounded reward for exercising actual policy updates."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # A deterministic bounded synthetic reward to exercise real policy updates.
    # This is a plumbing validation, not a math-quality evaluation.
    return float(sum(solution_str.encode("utf-8")) % 17) / 16.0
