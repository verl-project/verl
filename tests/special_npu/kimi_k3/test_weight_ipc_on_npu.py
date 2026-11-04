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
# ruff: noqa: E402

import pytest
import torch

pytest.importorskip("torch_npu")

from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_torch_device
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import rebuild_ipc


@pytest.mark.parametrize("source_device", ["cpu", "npu:0"])
def test_weight_ipc_rebuild_uses_receiver_npu(source_device):
    get_torch_device().set_device(0)
    source = torch.arange(16, dtype=torch.float32, device=source_device)
    rebuilt = rebuild_ipc(reduce_tensor(source), device_id=0)
    assert rebuilt.device == torch.device("npu:0")
    torch.testing.assert_close(rebuilt, source.to("npu:0"), rtol=0, atol=0)
