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
"""Per-request sampling seeds under rollout.full_determinism.

Before the fix every request served by one replica got ``replica_rank + seed``, so the n
samples of a GRPO group (same prompt, routed to the same replica) were identical and every
advantage was zero.
"""

import os
import subprocess
import sys

from verl.workers.rollout.utils import deterministic_request_seed


def _group_ids(num_prompts: int, n: int) -> list[list[str]]:
    # The agent loops use request_id = f"det-{priority}" with priority = arange(batch * n), and
    # the trainer repeats each prompt n times with interleave=True.
    return [[f"det-{p * n + k}" for k in range(n)] for p in range(num_prompts)]


def test_group_samples_get_distinct_seeds():
    for group in _group_ids(num_prompts=128, n=8):
        seeds = [deterministic_request_seed(42, rid) for rid in group]
        assert len(set(seeds)) == len(seeds), group


def test_all_requests_of_a_step_get_distinct_seeds():
    ids = [rid for group in _group_ids(num_prompts=512, n=16) for rid in group]
    seeds = {deterministic_request_seed(42, rid) for rid in ids}
    assert len(seeds) == len(ids)


def test_old_rule_collides():
    # Negative control: the replaced rule gives one seed to every request of a replica.
    replica_rank, base_seed = 3, 42
    old = {replica_rank + base_seed for _ in _group_ids(num_prompts=1, n=8)[0]}
    assert len(old) == 1


def test_seed_depends_on_base_seed_and_is_in_range():
    a = deterministic_request_seed(42, "det-0")
    b = deterministic_request_seed(43, "det-0")
    assert a != b
    for rid in ("det-0", "det-1", "x" * 1000, ""):
        s = deterministic_request_seed(2**40, rid)
        assert 0 <= s < 2**31 - 1


def test_seed_is_stable_across_processes():
    # Python's hash() is salted per process; the seed must not depend on it.
    code = "from verl.workers.rollout.utils import deterministic_request_seed as f;print(f(42, 'det-7'))"
    outs = set()
    for h in (0, 1):
        env = {**os.environ, "PYTHONHASHSEED": str(h)}
        outs.add(subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip())
    assert outs == {str(deterministic_request_seed(42, "det-7"))}
