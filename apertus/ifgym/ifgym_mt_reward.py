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
"""Per-turn IFGym reward for verl's reward manager.

NOTE: when using the IFGym agent loop (``default_agent_loop=ifgym_agent``),
per-turn scoring already happens during rollout and is carried via
``turn_scores`` / ``reward_score``; this reward function is then unused. It is
kept for the reward-manager / offline-eval path: it parses assistant turns from
``solution_str`` (ChatML special tokens preserved, or plain "assistant\\n..."
role markers if decoded with skip_special_tokens=True), scores each turn
against its active constraints, and returns the mean. Falls back to flat
scoring against the union of all constraints if no turn boundaries parse.

Set IFGYM_REWARD_DEBUG=1 for per-call breakdowns.
"""

import json
import os
import re

from apertus.ifgym.ifgym_instructions.instructions_registry import (
    HISTORY_AWARE_INSTRUCTIONS,
    INSTRUCTION_DICT,
)

_DEBUG = os.environ.get("IFGYM_REWARD_DEBUG") == "1"

# Format A: ChatML with special tokens preserved.
_RE_TOKENS_CLOSED = re.compile(r"<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>", re.DOTALL)
_RE_TOKENS_TRAILING = re.compile(r"<\|im_start\|>assistant\s*\n([^<]*)$", re.DOTALL)

# Format B: special tokens stripped (skip_special_tokens=True). Match
# "assistant\n<content>" up to next role marker or end-of-string.
_RE_PLAIN = re.compile(
    r"(?:^|\n)assistant\s*\n(.*?)(?=\n(?:user|system|assistant)\s*\n|\Z)",
    re.DOTALL,
)


def _split_assistant_turns(sol):
    closed = _RE_TOKENS_CLOSED.findall(sol)
    trailing = _RE_TOKENS_TRAILING.findall(sol)
    if closed or trailing:
        return closed + trailing, "im_tokens", bool(trailing)
    plain = _RE_PLAIN.findall(sol)
    if plain:
        return plain, "plain_role", False
    return [], "no_match", False


def _score_turn(response, constraints, prev_response):
    if not constraints:
        return 0.0
    n_checked = n_pass = 0
    for c in constraints:
        cid = c.get("constraint_id")
        kw = dict(c.get("kwargs") or {})
        if cid not in INSTRUCTION_DICT:
            continue
        if cid in HISTORY_AWARE_INSTRUCTIONS:
            kw["previous_response"] = prev_response
        try:
            inst = INSTRUCTION_DICT[cid](cid)
            inst.build_description(**kw)
            n_checked += 1
            if response and response.strip() and inst.check_following(response):
                n_pass += 1
        except Exception:
            continue
    return (n_pass / n_checked) if n_checked else 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if not extra_info:
        return 0.0
    ik = extra_info.get("interaction_kwargs") or {}
    turns_json = ik.get("turns_json")
    if not turns_json:
        return 0.0
    try:
        turns = json.loads(turns_json)
    except Exception:
        return 0.0

    sol = solution_str or ""
    responses, parser, clipped = _split_assistant_turns(sol)

    if not responses:
        # Score the whole blob against the union of all turns' constraints.
        # Worse signal than per-turn but non-degenerate.
        all_constraints = []
        for t in turns:
            all_constraints.extend(t.get("active_constraints", []))
        score = _score_turn(sol, all_constraints, None)
        if _DEBUG:
            print(
                f"[ifgym_reward] parser=no_match data_turns={len(turns)} fallback_score={score:.3f} sol_len={len(sol)}",
                flush=True,
            )
        return score

    turns_used = turns[: len(responses)]
    per_turn = []
    prev = None
    for i in range(len(turns_used)):
        per_turn.append(
            _score_turn(
                responses[i],
                turns_used[i].get("active_constraints", []),
                prev,
            )
        )
        prev = responses[i]
    score = sum(per_turn) / len(per_turn) if per_turn else 0.0

    if _DEBUG:
        print(
            f"[ifgym_reward] parser={parser} data_turns={len(turns)} "
            f"parsed_turns={len(responses)} clipped={clipped} "
            f"per_turn={[round(p, 2) for p in per_turn]} "
            f"mean={score:.3f}",
            flush=True,
        )
    return score
