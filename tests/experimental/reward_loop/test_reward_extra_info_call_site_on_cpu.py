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

"""Focused RewardLoopManager integration coverage without a Ray cluster."""

import torch

from verl.experimental.reward_loop import reward_loop as reward_loop_module
from verl.experimental.reward_loop.reward_loop import RewardLoopManager


class _FakeBatch:
    def __len__(self):
        return 2

    def chunk(self, num_chunks):
        assert num_chunks == 1
        return [self]


class _FakeRemoteCall:
    def __init__(self, outputs):
        self.outputs = outputs

    def remote(self, chunk):
        assert isinstance(chunk, _FakeBatch)
        return self.outputs


class _FakeWorker:
    def __init__(self, outputs):
        self.compute_score_batch = _FakeRemoteCall(outputs)


class _FakeRewardManager:
    @classmethod
    def assemble_rm_scores(cls, data, scores):
        assert isinstance(data, _FakeBatch)
        assert scores == [1.0, -1.0]
        return torch.zeros((len(data), 1), dtype=torch.float32)


def test_reward_loop_manager_uses_shared_reward_extra_info_assembly(monkeypatch):
    rich_info = {"score": 1.0, "acc": True, "pred": "42"}
    poor_info = {"acc": 0.0}
    outputs = [
        {"reward_score": 1.0, "reward_extra_info": rich_info},
        {"reward_score": -1.0, "reward_extra_info": poor_info},
    ]
    manager = type(
        "_FakeRewardLoopManager",
        (),
        {
            "reward_model_manager": None,
            "reward_loop_workers": [_FakeWorker(outputs)],
            "reward_manager_cls": _FakeRewardManager,
        },
    )()

    calls = []
    shared_assembler = reward_loop_module.assemble_reward_extra_info

    def assembly_spy(reward_extra_infos):
        calls.append(reward_extra_infos)
        return shared_assembler(reward_extra_infos)

    monkeypatch.setattr(reward_loop_module, "assemble_reward_extra_info", assembly_spy)
    monkeypatch.setattr(reward_loop_module.ray, "get", lambda pending: pending)
    monkeypatch.setattr(
        reward_loop_module,
        "pad_dataproto_to_divisor",
        lambda data, divisor: (data, 0),
    )

    result = RewardLoopManager.compute_rm_score(manager, _FakeBatch())

    assert calls == [[rich_info, poor_info]]
    assert result.meta_info["reward_extra_keys"] == ["score", "acc", "pred"]
    assert result.non_tensor_batch["score"].tolist() == [1.0, None]
    assert result.non_tensor_batch["acc"].tolist() == [1.0, 0.0]
    assert result.non_tensor_batch["pred"].tolist() == ["42", None]
    assert result.non_tensor_batch["score"].dtype == object
