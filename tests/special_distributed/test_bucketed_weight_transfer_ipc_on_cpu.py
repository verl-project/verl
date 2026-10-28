# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""CPU-only regression tests for vLLM weight-transfer IPC dispatch."""

import importlib.util
from pathlib import Path

import torch
from torch.multiprocessing.reductions import reduce_tensor

# IPC rebuilding itself has no vLLM dependency.
_path = Path(__file__).resolve().parents[2] / "verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"
_spec = importlib.util.spec_from_file_location("bucketed_weight_transfer_cpu", _path)
transfer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(transfer)


def test_rebuild_cpu_handle_does_not_inject_accelerator_device(monkeypatch):
    source = torch.arange(4, dtype=torch.float32)
    handle = reduce_tensor(source)
    monkeypatch.setattr(transfer, "get_device_name", lambda: "cpu")
    rebuilt = transfer.rebuild_ipc(handle, device_id=0)

    assert rebuilt.device.type == "cpu"
    torch.testing.assert_close(rebuilt, source)


def test_rebuild_accelerator_handle_uses_named_rebuilder(monkeypatch):
    captured = {}

    def rebuild_npu_tensor(*args):
        captured["args"] = args
        return torch.ones(1)

    monkeypatch.setattr(transfer, "get_device_name", lambda: "cpu")
    transfer.rebuild_ipc((rebuild_npu_tensor, (0, 1, 2, 3, 4, 5, 6)), device_id=9)

    assert captured["args"][6] == 9
