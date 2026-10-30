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
import weakref
from types import SimpleNamespace

import pytest
import torch

from verl.utils.fsdp_utils import materialize_fsdp_dtensor


@pytest.mark.parametrize("device_type", ["npu", "cuda", "cpu"])
def test_export_retains_staged_shard_until_npu_completion(monkeypatch, device_type):
    """Model a pending device write whose source must survive until completion."""
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output = torch.zeros_like(expected)
    device = SimpleNamespace(type=device_type)
    staged_ref = None
    waits = []

    class StagedShard:
        def __init__(self):
            self.device = device

        def full_tensor(self):
            if device_type != "npu":
                output.copy_(expected)
            return output

    class SourceShard:
        def to(self, destination, non_blocking):
            nonlocal staged_ref
            assert destination is device and non_blocking is True
            staged = StagedShard()
            staged_ref = weakref.ref(staged)
            return staged

    def complete(actual_device):
        assert actual_device is device
        assert staged_ref() is not None, "the pending transfer lost its staged shard"
        waits.append(actual_device)
        output.copy_(expected)

    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=complete), raising=False)
    result = materialize_fsdp_dtensor(SourceShard(), device)
    assert result is output
    torch.testing.assert_close(result, expected)
    assert waits == ([device] if device_type == "npu" else [])
    assert staged_ref() is None, "export should not retain completed staging buffers"
