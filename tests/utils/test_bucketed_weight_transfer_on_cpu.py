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


@pytest.mark.parametrize(
    ("sender_kwargs", "expected_setting", "expected_point"),
    [
        ({"gc_diagnostics": True}, True, "weight_transfer_cleanup"),
        ({"gc_on_cleanup": False}, False, None),
        ({"gc_on_cleanup": 0}, 0, None),
        ({"gc_on_cleanup": 1}, 1, None),
    ],
)
def test_sender_cleanup_forwards_gc_configuration(monkeypatch, sender_kwargs, expected_setting, expected_point):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    sender = bucketed_weight_transfer.BucketedWeightSender(
        zmq_handle="tcp://unused",
        bucket_size_mb=1,
        use_shm=True,
        **sender_kwargs,
    )
    calls = []
    monkeypatch.setattr(
        bucketed_weight_transfer,
        "collect_garbage",
        lambda setting, **kwargs: calls.append((setting, kwargs)),
    )
    monkeypatch.setattr(bucketed_weight_transfer, "get_torch_device", lambda: _FakeTorchDevice())

    sender._cleanup()

    assert calls == [(expected_setting, {"diagnostics_point": expected_point})]
