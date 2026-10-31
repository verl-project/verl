# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import patch

import pytest

from verl.utils.profiler.config import ProfilerConfig
from verl.utils.profiler.profile import DistProfiler, build_rollout_dist_profiler


@pytest.mark.parametrize("selected", [True, False])
def test_init_uses_rank_selection_result(selected):
    config = ProfilerConfig(enable=True, tool=None, tool_config=None)
    with patch("verl.utils.profiler.profile._profile_rank_selected", return_value=selected) as select_rank:
        profiler = DistProfiler(rank=0, config=config)
    assert profiler.check_this_rank() is selected
    select_rank.assert_called_once_with(config, 0)


class TestBuildRolloutDistProfiler:
    @patch("verl.utils.profiler.profile.DistProfiler")
    @patch("verl.utils.profiler.profile._rollout_replica_ranks", return_value=[1])
    def test_mapped_ranks_are_used_without_mutating_source_config(self, rank_mapper, profiler):
        config = ProfilerConfig(ranks=[8])
        result = build_rollout_dist_profiler(1, 8, config=config)

        rank_mapper.assert_called_once_with([8], 8)
        mapped_config = profiler.call_args.kwargs["config"]
        assert mapped_config.ranks == [1]
        assert config.ranks == [8]
        assert mapped_config is not config
        profiler.assert_called_once_with(rank=1, config=mapped_config, tool_config=None)
        assert result is profiler.return_value

    @pytest.mark.parametrize(
        "config, rank",
        [(ProfilerConfig(all_ranks=True, ranks=[0]), 2), (ProfilerConfig(ranks=[]), 0)],
        ids=["all-ranks", "empty-ranks"],
    )
    @patch("verl.utils.profiler.profile.DistProfiler")
    @patch("verl.utils.profiler.profile._rollout_replica_ranks", return_value=[])
    def test_rank_selection_config_is_preserved(self, rank_mapper, profiler, config, rank):
        build_rollout_dist_profiler(rank, 1, config=config)
        passed_config = profiler.call_args.kwargs["config"]
        assert passed_config.all_ranks == config.all_ranks
        assert passed_config.ranks == config.ranks
        profiler.assert_called_once_with(rank=rank, config=passed_config, tool_config=None)
