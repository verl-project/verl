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

from verl.utils.reward_score.prime_code import utils


def test_check_correctness_returns_plain_metadata_list():
    sample = {"inputs": [[1]], "outputs": [[1]], "fn_name": "solve"}
    generation = "def solve(x):\n    return x"

    result, metadata = utils.check_correctness(sample, generation, timeout=1, debug=False)

    assert type(result) is list
    assert type(metadata) is list


def test_timeout_reaps_worker_and_shuts_down_manager(monkeypatch):
    managers = []
    processes = []

    class FakeManager:
        def __init__(self):
            self.entered = False
            self.exited = False
            managers.append(self)

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.exited = True

        def list(self):
            return []

    class FakeProcess:
        def __init__(self, target, args):
            self.join_calls = []
            self.killed = False
            processes.append(self)

        def start(self):
            pass

        def join(self, timeout=None):
            self.join_calls.append(timeout)

        def is_alive(self):
            return not self.killed

        def kill(self):
            self.killed = True

    monkeypatch.setattr(utils.multiprocessing, "Manager", FakeManager)
    monkeypatch.setattr(utils.multiprocessing, "Process", FakeProcess)

    sample = {"inputs": [[1]], "outputs": [[1]]}
    result, metadata = utils.check_correctness(sample, "while True: pass", timeout=0.1, debug=False)

    assert result == [-1]
    assert metadata == []
    assert managers[0].entered
    assert managers[0].exited
    assert processes[0].killed
    assert len(processes[0].join_calls) == 2
    assert processes[0].join_calls[0] == pytest.approx(1.1)
    assert processes[0].join_calls[1] is None
