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

import sys
import types
from unittest.mock import MagicMock, call, patch

from verl.utils.tracking import DapoFilteredRewardTableLogger, Tracking, ValidationGenerationsLogger


def test_tracking_finish_finalizes_wandb_once():
    tracking = Tracking.__new__(Tracking)
    tracking.logger = {"wandb": MagicMock()}
    tracking._finished = False

    tracking.finish(exit_code=1)
    tracking.finish(exit_code=0)

    tracking.logger["wandb"].finish.assert_called_once_with(exit_code=1)


def test_dapo_filtered_reward_table_logs_incremental_rows():
    mock_wandb = MagicMock()
    mock_wandb.run = object()
    mock_wandb.__version__ = "0.20.0"
    table = mock_wandb.Table.return_value
    logger = DapoFilteredRewardTableLogger()

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        logger.log(["wandb"], {1.0: 2, 0.0: 3}, step=7)
        logger.log(["wandb"], {0.0: 4}, step=8)

    mock_wandb.Table.assert_called_once_with(columns=["step", "reward_counts"], log_mode="INCREMENTAL")
    assert table.add_data.call_args_list == [call(7, "0:3, 1:2"), call(8, "0:4")]
    assert mock_wandb.log.call_args_list == [
        call({"training/filter_groups/filtered_reward_counts": table}, step=7),
        call({"training/filter_groups/filtered_reward_counts": table}, step=8),
    ]


def test_dapo_filtered_reward_table_falls_back_to_full_history():
    mock_wandb = MagicMock()
    mock_wandb.run = object()
    mock_wandb.__version__ = "0.19.11"
    first_table, second_table = MagicMock(), MagicMock()
    mock_wandb.Table.side_effect = [first_table, second_table]
    logger = DapoFilteredRewardTableLogger()

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        logger.log(["wandb"], {1.0: 2, 0.0: 3}, step=7)
        logger.log(["wandb"], {0.0: 4}, step=8)

    assert mock_wandb.Table.call_args_list == [
        call(columns=["step", "reward_counts"], data=[[7, "0:3, 1:2"]]),
        call(columns=["step", "reward_counts"], data=[[7, "0:3, 1:2"], [8, "0:4"]]),
    ]
    assert mock_wandb.log.call_args_list == [
        call({"training/filter_groups/filtered_reward_counts": first_table}, step=7),
        call({"training/filter_groups/filtered_reward_counts": second_table}, step=8),
    ]


def test_validation_generations_logger_logs_trackio_traces():
    mock_trackio = MagicMock()
    mock_trackio.context_vars = types.SimpleNamespace(current_run=MagicMock())
    mock_trackio.context_vars.current_run.get.return_value = None
    mock_trackio.Trace.side_effect = lambda messages, metadata=None: {
        "_type": "trackio.trace",
        "messages": messages,
        "metadata": metadata or {},
    }

    with patch.dict(sys.modules, {"trackio": mock_trackio}):
        ValidationGenerationsLogger().log(
            ["trackio"],
            samples=[["question", "answer", 0.5]],
            step=7,
        )

    mock_trackio.Trace.assert_called_once()
    trace_kwargs = mock_trackio.Trace.call_args.kwargs
    assert trace_kwargs["messages"] == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert trace_kwargs["metadata"]["source"] == "validation_generations"
    assert trace_kwargs["metadata"]["score"] == 0.5
    mock_trackio.log.assert_called_once()
    assert mock_trackio.log.call_args.kwargs["step"] == 7
