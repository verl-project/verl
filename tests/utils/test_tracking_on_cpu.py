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
from unittest.mock import MagicMock, patch

from verl.utils.tracking import ValidationGenerationsLogger


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
