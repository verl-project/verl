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

import ast
import json
from collections.abc import Mapping

from .np_rules import (
    GCP_D,
    TSP,
    hamiltonian_cycle,
    knapsack,
    maximum_clique,
    maximum_set,
    meeting_schedule,
    minimum_cut,
    set_cover,
    subset_sum,
)

NP_DATA_SOURCE_TO_VALIDATOR = {
    "bootcamp/NpGcpD": GCP_D.validation,
    "bootcamp/NpHamiltonianCycle": hamiltonian_cycle.validation,
    "bootcamp/NpMaximumCliqueProblem": maximum_clique.validation,
    "bootcamp/NpMaximumSet": maximum_set.validation,
    "bootcamp/NpMinimumCut": minimum_cut.validation,
    "bootcamp/NpTsp": TSP.validation,
    "bootcamp/NpKnapsack": knapsack.validation,
    "bootcamp/NpSetCover": set_cover.validation,
    "bootcamp/NpSubsetSum": subset_sum.validation,
    "bootcamp/NpMeetingSchedule": meeting_schedule.validation,
}

LEGACY_TASK_TO_VALIDATOR = {
    "GCP-D": GCP_D.validation,
    "TSP": TSP.validation,
    "hamiltonian-cycle": hamiltonian_cycle.validation,
    "knapsack": knapsack.validation,
    "maximum_clique_problem": maximum_clique.validation,
    "maximum-set": maximum_set.validation,
    "meeting-schedule": meeting_schedule.validation,
    "minimum-cut": minimum_cut.validation,
    "set-cover": set_cover.validation,
    "subset-sum": subset_sum.validation,
}


def _parse_structured_value(value):
    if not isinstance(value, str):
        return value

    value = value.strip()
    if not value:
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def _resolve_payload(ground_truth, extra_info):
    payload = _parse_structured_value(ground_truth)
    if isinstance(payload, Mapping):
        question = payload.get("question")
        target = payload.get("ground_truth")
    else:
        question = extra_info.get("question")
        target = payload

    question = _parse_structured_value(question)
    target = _parse_structured_value(target)
    if question is None:
        raise ValueError("NP reward requires a question in ground_truth or extra_info.")
    return question, target


def _resolve_validator(data_source, extra_info):
    if data_source in NP_DATA_SOURCE_TO_VALIDATOR:
        return NP_DATA_SOURCE_TO_VALIDATOR[data_source]

    task_type = extra_info.get("task_type") or extra_info.get("type") or extra_info.get("split")
    if task_type in LEGACY_TASK_TO_VALIDATOR:
        return LEGACY_TASK_TO_VALIDATOR[task_type]

    raise ValueError(f"Unknown NP data source or task type: {data_source!r}")


def compute_score(
    solution_str: str,
    ground_truth,
    extra_info: dict | None = None,
    data_source: str | None = None,
    format_reward: float = -1.0,
    answer_reward: float = -1.5,
):
    """Compute the rule-based reward for Bootcamp NP tasks."""
    extra_info = extra_info or {}
    validator = _resolve_validator(data_source, extra_info)
    question, target = _resolve_payload(ground_truth, extra_info)

    has_answer_tag = "Answer:" in solution_str
    is_invalid, score, _message = validator(question, solution_str, target)

    format_score = 1.0 if has_answer_tag else format_reward
    answer_score = answer_reward if is_invalid else score
    return float(format_score + answer_score)
