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

import torch
from tensordict import TensorDict

from verl.trainer.ppo.v1.utils import rollout_logprob_dump_fields


def nested(rows):
    return torch.nested.as_nested_tensor([torch.tensor(r) for r in rows], layout=torch.jagged)


def test_rollout_logprob_dump_fields():
    data = TensorDict(
        {
            "prompts": nested([[1, 2, 3], [4]]),
            "old_log_probs": nested([[-0.5, -1.0], [-0.25, -2.0, -3.0]]),
            "rollout_log_probs": nested([[-0.5, -1.5], [-0.25, -2.0, -3.5]]),
        },
        batch_size=[2],
    )
    out = rollout_logprob_dump_fields(data)
    assert out["old_log_probs"] == [[-0.5, -1.0], [-0.25, -2.0, -3.0]]
    assert out["rollout_log_probs"] == [[-0.5, -1.5], [-0.25, -2.0, -3.5]]
    assert out["prompt_len"] == [3, 1]


def test_rollout_logprob_dump_fields_without_rollout_log_probs():
    data = TensorDict({"prompts": nested([[1], [2, 3]]), "old_log_probs": nested([[-1.0], [-2.0]])}, batch_size=[2])
    out = rollout_logprob_dump_fields(data)
    assert set(out) == {"old_log_probs", "prompt_len"}
