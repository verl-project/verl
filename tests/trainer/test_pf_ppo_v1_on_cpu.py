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
"""PF-PPO reweighting resamples rows; the v1 advantage path must not do so silently.

The v1 trainer reads a batch out of the TransferQueue, computes advantages, and writes
``advantages``/``returns`` back under the *same* sample keys it read
(``verl/trainer/ppo/v1/trainer_base.py``, ``kv_batch_put(keys=batch.keys, ...)``).
That write-back is only correct while row ``i`` of the computed advantages still belongs
to key ``i``. ``compute_pf_ppo_reweight_data`` resamples rows with replacement, which
breaks exactly that invariant, so the v1 path must reject the config instead of
returning rows that no longer line up with the keys.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_pf_ppo_reweight_data
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories

BATCH_SIZE = 8
RESPONSE_LEN = 4


def _pf_ppo_config() -> OmegaConf:
    return OmegaConf.create({"use_pf_ppo": True, "pf_ppo": {"reweight_method": "pow", "weight_pow": 2.0}})


def _batch() -> DataProto:
    """One row per uid, with a distinct score per row so resampling is observable."""
    scores = torch.arange(BATCH_SIZE, dtype=torch.float32).unsqueeze(-1).repeat(1, RESPONSE_LEN)
    tensors = TensorDict(
        {
            "token_level_scores": scores.clone(),
            "token_level_rewards": scores.clone(),
            "values": torch.zeros(BATCH_SIZE, RESPONSE_LEN),
            "response_mask": torch.ones(BATCH_SIZE, RESPONSE_LEN),
        },
        batch_size=[BATCH_SIZE],
    )
    uids = np.array([f"uid{i}" for i in range(BATCH_SIZE)], dtype=object)
    return DataProto(batch=tensors, non_tensor_batch={"uid": uids})


def _batch_keys() -> list[str]:
    # v1 keys are "{uid}_{session_id}_{index}"
    return [f"uid{i}_0_0" for i in range(BATCH_SIZE)]


def test_pf_ppo_reweight_does_not_preserve_row_identity():
    """Documents why the v1 path cannot carry PF-PPO: rows are resampled with replacement."""
    data = _batch()
    torch.manual_seed(0)
    out = compute_pf_ppo_reweight_data(data, "pow", 2.0)

    assert list(out.non_tensor_batch["uid"]) != [f"uid{i}" for i in range(BATCH_SIZE)], (
        "expected PF-PPO to resample rows; if this ever holds, the v1 rejection can be revisited"
    )
    # every surviving row is still internally consistent, it just is not row i any more
    for row, uid in enumerate(out.non_tensor_batch["uid"]):
        source = int(uid.removeprefix("uid"))
        assert out.batch["token_level_scores"][row, 0].item() == pytest.approx(float(source))


def test_use_pf_ppo_is_rejected_on_v1_advantage_path():
    """v1 must fail loudly rather than write resampled advantages back under the original keys."""
    with pytest.raises(NotImplementedError, match="use_pf_ppo"):
        compute_advantage_for_multi_trajectories(
            data=_batch(),
            batch_keys=_batch_keys(),
            adv_estimator=AdvantageEstimator.GAE,
            config=_pf_ppo_config(),
        )


def test_v1_advantage_path_keeps_rows_aligned_with_batch_keys_without_pf_ppo():
    """The invariant the v1 write-back depends on still holds when PF-PPO is off."""
    out = compute_advantage_for_multi_trajectories(
        data=_batch(),
        batch_keys=_batch_keys(),
        adv_estimator=AdvantageEstimator.GAE,
        config=OmegaConf.create({"use_pf_ppo": False, "pf_ppo": {}}),
    )
    assert list(out.non_tensor_batch["uid"]) == [f"uid{i}" for i in range(BATCH_SIZE)]


def test_v0_compute_advantage_still_applies_pf_ppo():
    """The v0 trainer resamples the whole batch and carries it forward, so it keeps PF-PPO."""
    torch.manual_seed(0)
    out = compute_advantage(_batch(), adv_estimator=AdvantageEstimator.GAE, config=_pf_ppo_config())
    assert list(out.non_tensor_batch["uid"]) != [f"uid{i}" for i in range(BATCH_SIZE)]
