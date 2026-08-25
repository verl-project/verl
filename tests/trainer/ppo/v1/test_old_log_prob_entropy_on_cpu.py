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

from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

pytest.importorskip("ray")
pytest.importorskip("transfer_queue")

from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _Batch:
    def __init__(self):
        self.keys = ["sample-0", "sample-1"]
        self.partition_id = "train"
        self.extra_info = {}

    def __len__(self):
        return len(self.keys)


class _ActorRolloutWorkerGroup:
    def compute_log_prob(self, batch):
        return batch


@pytest.mark.parametrize(
    ("calculate_entropy", "entropy_coeff", "expected"),
    [
        (False, 0.0, False),
        (True, 0.0, True),
        (False, 0.01, True),
        (True, 0.01, True),
    ],
)
def test_compute_old_log_prob_respects_entropy_config(calculate_entropy, entropy_coeff, expected):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "calculate_entropy": calculate_entropy,
                    "entropy_coeff": entropy_coeff,
                    "loss_agg_mode": "token-mean",
                    "loss_scale_factor": None,
                },
                "rollout": {"temperature": 1.0, "calculate_log_probs": False},
            },
            "algorithm": {"rollout_correction": None},
        }
    )
    trainer.actor_rollout_wg = _ActorRolloutWorkerGroup()
    batch = _Batch()
    data = TensorDict(
        {
            "log_probs": torch.nested.as_nested_tensor(
                [torch.tensor([10.0, 11.0, 12.0, 13.0]), torch.tensor([20.0, 21.0, 22.0, 23.0])],
                layout=torch.jagged,
            ),
            "response_mask": torch.nested.as_nested_tensor(
                [torch.ones(2), torch.ones(2)],
                layout=torch.jagged,
            ),
        },
        batch_size=[2],
    )
    if expected:
        data["entropy"] = torch.nested.as_nested_tensor(
            [torch.full((4,), 3.0), torch.full((4,), 3.0)],
            layout=torch.jagged,
        )
    metrics = {}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get", return_value=data) as kv_batch_get,
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_put", return_value=batch) as kv_batch_put,
    ):
        result = trainer._compute_old_log_prob(batch, metrics)

    assert result is batch
    assert batch.extra_info["calculate_entropy"] is expected
    expected_fields = ["log_probs", "response_mask", *(["entropy"] if expected else [])]
    assert kv_batch_get.call_args.kwargs["select_fields"] == expected_fields
    fields = kv_batch_put.call_args.kwargs["fields"]
    assert torch.equal(fields["old_log_probs"].values(), torch.tensor([11.0, 12.0, 21.0, 22.0]))
    expected_entropy = torch.full((4,), 3.0) if expected else torch.zeros(4)
    assert torch.equal(fields["entropy"].values(), expected_entropy)
    assert metrics["actor/entropy"] == (3.0 if expected else 0.0)
