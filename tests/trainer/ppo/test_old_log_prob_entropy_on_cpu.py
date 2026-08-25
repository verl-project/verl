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

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("ray")

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu


class _ActorRolloutWorkerGroup:
    def __init__(self):
        self.calculate_entropy = None

    def compute_log_prob(self, batch):
        self.calculate_entropy = tu.get_non_tensor_data(batch, "calculate_entropy", None)
        output = {
            "log_probs": torch.nested.as_nested_tensor(
                [torch.tensor([10.0, 11.0, 12.0, 13.0]), torch.tensor([20.0, 21.0, 22.0, 23.0])],
                layout=torch.jagged,
            )
        }
        if self.calculate_entropy:
            output["entropy"] = torch.nested.as_nested_tensor(
                [torch.full((4,), 3.0), torch.full((4,), 3.0)],
                layout=torch.jagged,
            )
        output = tu.get_tensordict(output)
        tu.assign_non_tensor(output, metrics={"mfu": 0.5})
        return output


def _make_batch():
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2], [3, 4]]),
            "responses": torch.tensor([[5, 6], [7, 8]]),
            "input_ids": torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]]),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "response_mask": torch.ones(2, 2, dtype=torch.long),
            "position_ids": torch.arange(4).repeat(2, 1),
        }
    )


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
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "calculate_entropy": calculate_entropy,
                    "entropy_coeff": entropy_coeff,
                    "calculate_sum_pi_squared": False,
                }
            }
        }
    )
    trainer.actor_rollout_wg = _ActorRolloutWorkerGroup()

    old_log_prob, old_log_prob_mfu = trainer._compute_old_log_prob(_make_batch())

    assert trainer.actor_rollout_wg.calculate_entropy is expected
    assert old_log_prob_mfu == 0.5
    assert torch.equal(old_log_prob.batch["old_log_probs"], torch.tensor([[11.0, 12.0], [21.0, 22.0]]))
    expected_entropy = torch.full((2, 2), 3.0) if expected else torch.zeros(2, 2)
    assert torch.equal(old_log_prob.batch["entropys"], expected_entropy)
