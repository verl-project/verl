# Copyright 2026 Individual Contributor
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
"""Trainer wiring of ``kl_loss_type: topk_rb`` in the PPO trainer, with stub worker groups.

Checks that the reference pass requests ``ref_topk``, that its ``ref_topk_log_probs`` / ``ref_topk_ids``
outputs reach the batch aligned with the token stream, that the actor update runs with ``kl_topk_rb``, and
that ``ref_topk`` is cleared after the reference pass, since ``extra_info`` persists across calls.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding

K = 3


def _config(topk_rb: bool):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": topk_rb,
                    "kl_loss_type": "topk_rb" if topk_rb else "low_var_kl",
                    "kl_topk": K,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "data_loader_seed": 0,
                    "shuffle": False,
                    "policy_loss": {},
                },
                "rollout": {"n": 1, "temperature": 1.0, "multi_turn": {"enable": False}},
            },
            "algorithm": {},
        }
    )


def _padded_batch():
    # two sequences, left-padded prompts (3 max) and right-padded responses (2 max)
    prompts = torch.tensor([[0, 5, 6], [7, 8, 9]])
    responses = torch.tensor([[1, 2], [3, 0]])
    attn = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    return TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], 1),
            "attention_mask": attn,
            "position_ids": (attn.cumsum(-1) - 1).clamp(min=0),
            "response_mask": torch.tensor([[1, 1], [1, 0]]),
        },
        batch_size=[2],
    )


def _fake_ref_outputs(td):
    """What the ref engine returns: nested full-sequence outputs; values encode the packed row index."""
    cu = td["input_ids"].offsets()
    n = int(cu[-1])
    rows = torch.arange(n, dtype=torch.float32)
    return tu.get_tensordict(
        {
            "log_probs": torch.nested.nested_tensor_from_jagged(-rows, offsets=cu),
            "ref_topk_log_probs": torch.nested.nested_tensor_from_jagged(
                -(rows.unsqueeze(-1) + torch.arange(K + 1) / 10), offsets=cu
            ),
            "ref_topk_ids": torch.nested.nested_tensor_from_jagged(
                (rows.unsqueeze(-1) * 10 + torch.arange(K)).to(torch.int32), offsets=cu
            ),
        }
    )


class _FakeTQ:
    """In-memory stand-in for transfer_queue's kv_batch_get / kv_batch_put."""

    def __init__(self, td):
        self.store = td.clone()
        # v1 keeps per-sample fields nested (jagged) in TransferQueue
        mask = td["response_mask"]
        self.store["response_mask"] = torch.nested.as_nested_tensor(
            [row[: int(row.sum())] for row in mask], layout=torch.jagged
        )

    def kv_batch_get(self, keys, partition_id, select_fields):
        return self.store.select(*select_fields)

    def kv_batch_put(self, keys, partition_id, fields):
        for k in fields.keys():
            self.store[k] = fields[k]


class _Meta(SimpleNamespace):
    def __len__(self):
        return 2


@pytest.mark.parametrize("topk_rb", [True, False])
def test_v1_ref_pass_and_actor_update(topk_rb):
    from verl.trainer.ppo.v1 import trainer_base

    td = left_right_2_no_padding(_padded_batch())
    fake_tq = _FakeTQ(td)
    meta = _Meta(keys=["a", "b"], partition_id="p", extra_info={})
    seen = {}

    class RefWG:
        def compute_ref_log_prob(self, batch):
            seen["ref_topk"] = batch.extra_info.get("ref_topk")
            for k, v in _fake_ref_outputs(td).items():  # the worker writes its outputs to TransferQueue
                if seen["ref_topk"] or not k.startswith("ref_topk"):
                    fake_tq.store[k] = v
            return batch

    class ActorWG:
        def update_actor(self, batch):
            seen["actor_extra"] = dict(batch.extra_info)
            return {"metrics": {"mfu": [0.0]}}

    stub = SimpleNamespace(
        config=_config(topk_rb),
        ref_in_actor=False,
        ref_policy_wg=RefWG(),
        actor_rollout_wg=ActorWG(),
        distillation_config=None,
    )
    with patch.object(trainer_base, "tq", fake_tq):
        trainer_base.PPOTrainer._compute_ref_log_prob(stub, meta, metrics={})
        trainer_base.PPOTrainer._update_actor(stub, meta, metrics={})

    assert seen["ref_topk"] == (K if topk_rb else 0)
    assert meta.extra_info["ref_topk"] == 0  # cleared right after the reference pass
    assert seen["actor_extra"]["kl_topk_rb"] is topk_rb
    assert seen["actor_extra"]["ref_topk"] == 0
    if topk_rb:
        expected = _fake_ref_outputs(td)
        torch.testing.assert_close(fake_tq.store["ref_topk_ids"].values(), expected["ref_topk_ids"].values())
        torch.testing.assert_close(
            fake_tq.store["ref_topk_log_probs"].values(), expected["ref_topk_log_probs"].values()
        )
    else:
        assert "ref_topk_ids" not in fake_tq.store.keys()
