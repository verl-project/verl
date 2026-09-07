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

import pytest

pytest.importorskip("vllm")


class _FakeTorchDevice:
    def empty_cache(self):
        pass


def test_sender_cleanup_reports_gc_diagnostics_when_enabled(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    sender = bucketed_weight_transfer.BucketedWeightSender(
        zmq_handle="tcp://unused",
        bucket_size_mb=1,
        use_shm=True,
        gc_diagnostics=True,
    )
    calls = []
    monkeypatch.setattr(
        bucketed_weight_transfer,
        "collect_garbage",
        lambda setting, **kwargs: calls.append((setting, kwargs)),
    )
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    sender._cleanup()

    assert calls == [(True, {"diagnostics_point": "weight_transfer_cleanup"})]


@pytest.mark.parametrize("gc_setting", [False, 0, 1])
def test_sender_cleanup_forwards_gc_setting(monkeypatch, gc_setting):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    sender = bucketed_weight_transfer.BucketedWeightSender(
        zmq_handle="tcp://unused",
        bucket_size_mb=1,
        use_shm=True,
        gc_on_cleanup=gc_setting,
    )
    calls = []
    monkeypatch.setattr(
        bucketed_weight_transfer,
        "collect_garbage",
        lambda setting, **kwargs: calls.append((setting, kwargs)),
    )
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    sender._cleanup()

    assert calls == [(gc_setting, {"diagnostics_point": None})]
