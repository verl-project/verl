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
"""Test the distributed correctness probe's exit contract, not GPU collectives."""

from unittest.mock import Mock

import pytest

from tests.special_distributed import test_sharded_delta_gather as probe


@pytest.mark.parametrize(
    "world,case_count,rank", [(1, 4, 0), (2, 7, 0), (2, 7, 1), (4, 15, 0), (4, 15, 1), (8, 23, 0), (8, 23, 1)]
)
@pytest.mark.parametrize("failure_position", [None, "first", "middle", "last"])
def test_probe_exit_after_all_cases_and_teardown(monkeypatch, capsys, world, case_count, rank, failure_position):
    """A failed comparison must survive later passes and be observable by torchrun."""
    is_comparing_rank = rank == 0
    events = []
    monkeypatch.setattr(probe.dist, "init_process_group", Mock())
    monkeypatch.setattr(probe.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(probe.dist, "get_world_size", lambda: world)
    monkeypatch.setattr(probe.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(probe, "init_device_mesh", Mock())
    monkeypatch.setattr(probe.dist, "barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(probe.dist, "destroy_process_group", lambda: events.append("destroy"))
    failed_index = {None: None, "first": 0, "middle": case_count // 2, "last": case_count - 1}[failure_position]

    def run_case(*args):
        index = len(events)
        events.append("case")
        return not (is_comparing_rank and index == failed_index)

    monkeypatch.setattr(probe, "_run_case", run_case)
    code = probe.main()

    assert events == ["case"] * case_count + ["barrier", "destroy"]
    expected_failure = is_comparing_rank and failure_position is not None
    # Falling through main() is exit 0, just as in the uncorrected CLI.
    assert (0 if code is None else code) == int(expected_failure)
    output = capsys.readouterr().out
    if is_comparing_rank:
        assert ("OVERALL: FAIL" if expected_failure else "OVERALL: ALL PASS") in output
    else:
        assert not output
