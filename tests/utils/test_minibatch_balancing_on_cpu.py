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

import random
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import get_balance_mini_batch_size
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_minibatch_balanced_partitions,
    get_seqlen_balanced_partitions,
)


def _worker_mini_batches(partitions, mini_batch_size):
    """Mimic workers: each rank slices its shard into mini-batches sequentially."""
    share = mini_batch_size // len(partitions)
    num_mini_batches = len(partitions[0]) // share
    return [[partition[i * share : (i + 1) * share] for partition in partitions] for i in range(num_mini_batches)]


def _check_layout(partitions, num_samples, k_partitions):
    """Every sample appears exactly once and all ranks get the same number of samples."""
    assert len(partitions) == k_partitions
    assert sorted(i for partition in partitions for i in partition) == list(range(num_samples))
    assert len({len(partition) for partition in partitions}) == 1


def test_balances_each_mini_batch():
    """Core correctness: balancing the whole batch can still leave stragglers inside a mini-batch.

    Keeping each mini-batch balanced guarantees that every sequential update on every DP rank
    processes roughly the same number of tokens, which is the whole point of this PR.
    """
    workloads = [10, 10, 1, 1, 20, 20, 2, 2]

    partitions = get_minibatch_balanced_partitions(workloads, k_partitions=2, mini_batch_size=4)

    _check_layout(partitions, 8, 2)
    for shares in _worker_mini_batches(partitions, mini_batch_size=4):
        sums = [sum(workloads[j] for j in share) for share in shares]
        assert sums[0] == sums[1]


def test_shuffle_is_seeded_and_keeps_mini_batches_aligned():
    """Trainer-side shuffle must be deterministic across DP ranks.

    All ranks visit the same global mini-batch at each step; otherwise gradients would be wrong
    and the step could hang. This test also checks that the shuffle actually changes the order.
    """
    num_samples, mini_batch_size, k_partitions = 48, 12, 3
    workloads = list(range(1, num_samples + 1))

    partitions = get_minibatch_balanced_partitions(workloads, k_partitions, mini_batch_size, shuffle=True, seed=[42, 7])

    _check_layout(partitions, num_samples, k_partitions)
    # Same seed -> same partition layout.
    assert partitions == get_minibatch_balanced_partitions(
        workloads, k_partitions, mini_batch_size, shuffle=True, seed=[42, 7]
    )
    # Different seed -> different layout.
    assert partitions != get_minibatch_balanced_partitions(
        workloads, k_partitions, mini_batch_size, shuffle=True, seed=[42, 8]
    )
    mini_batches = [sorted(j for share in shares for j in share) for shares in _worker_mini_batches(partitions, 12)]
    assert all(len(members) == mini_batch_size for members in mini_batches)
    assert mini_batches != [list(range(i * 12, (i + 1) * 12)) for i in range(4)]


def _config(actor_shuffle=False, critic_shuffle=False, actor_epochs=1, critic_epochs=1):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"n": 2},
                "actor": {"ppo_mini_batch_size": 4, "shuffle": actor_shuffle, "ppo_epochs": actor_epochs},
            },
            "critic": {"ppo_mini_batch_size": 6, "shuffle": critic_shuffle, "ppo_epochs": critic_epochs},
        }
    )


def test_balance_mini_batch_size():
    """The balancing unit must divide every active PPO mini-batch size.

    Enabling ``shuffle`` with multiple PPO epochs on an active model disables per-mini-batch
    balancing because workers reshuffle samples inside each DP rank across epochs.
    Returns 0 when no model is updated.
    """
    config = _config()
    assert get_balance_mini_batch_size(config, use_critic=False, update_actor=True) == 8
    assert get_balance_mini_batch_size(config, use_critic=True, update_actor=False) == 12
    assert get_balance_mini_batch_size(config, use_critic=True, update_actor=True) == 4
    assert get_balance_mini_batch_size(config, use_critic=False, update_actor=False) == 0
    # Single-epoch shuffle and multiple epochs without shuffle retain per-mini-batch balancing.
    assert get_balance_mini_batch_size(_config(actor_shuffle=True), use_critic=False, update_actor=True) == 8
    assert get_balance_mini_batch_size(_config(critic_shuffle=True), use_critic=True, update_actor=True) == 4
    assert get_balance_mini_batch_size(_config(actor_epochs=2), use_critic=False, update_actor=True) == 8
    assert get_balance_mini_batch_size(_config(critic_epochs=2), use_critic=True, update_actor=False) == 12
    # Only active models that shuffle over multiple epochs trigger whole-batch balancing.
    actor_reshuffle = _config(actor_shuffle=True, actor_epochs=2)
    assert get_balance_mini_batch_size(actor_reshuffle, use_critic=False, update_actor=True) == 0
    assert get_balance_mini_batch_size(actor_reshuffle, use_critic=True, update_actor=False) == 12
    reshuffle = _config(critic_shuffle=True, critic_epochs=2)
    assert get_balance_mini_batch_size(reshuffle, use_critic=True, update_actor=True) == 0
    assert get_balance_mini_batch_size(reshuffle, use_critic=False, update_actor=True) == 8


@pytest.mark.parametrize("shuffle, epochs", [(False, 2), (True, 1), (True, 2)])
def test_ray_trainer_balance_batch(shuffle, epochs):
    """End-to-end check for the legacy trainer.

    Covers three cases:
    1. shuffle=False, epochs>1: per-mini-batch balance, deterministic order.
    2. shuffle=True, epochs=1: per-mini-batch balance, seeded shuffle.
    3. shuffle=True, epochs>1: fall back to whole-batch balance (original behavior).
    """
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    random.seed(0)
    num_samples, k_partitions = 32, 2
    lengths = [random.randint(10, 100) for _ in range(num_samples)]
    attention_mask = torch.zeros(num_samples, 128, dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    batch = DataProto.from_single_dict({"attention_mask": attention_mask, "index": torch.arange(num_samples)})

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = _config(actor_shuffle=shuffle, actor_epochs=epochs)
    trainer.config.trainer = {"balance_batch": True, "critic_warmup": 0}
    trainer.config.actor_rollout_ref.actor.data_loader_seed = 42
    trainer.use_critic, trainer.global_steps = False, 1
    trainer.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": list(range(k_partitions))})
    trainer._balance_batch(batch, metrics={})

    order = batch.batch["index"].tolist()
    shards = [order[: num_samples // k_partitions], order[num_samples // k_partitions :]]
    if shuffle and epochs > 1:
        # Whole-batch fallback: layout matches the old equal-size partition.
        assert not batch.meta_info["minibatch_balanced"]
        expected = get_seqlen_balanced_partitions(calculate_workload(torch.tensor(lengths)), 2, equal_size=True)
        assert [sorted(shard) for shard in shards] == [sorted(partition) for partition in expected]
        return

    # Per-mini-batch balance.
    assert batch.meta_info["minibatch_balanced"]
    mini_batches = [sorted(j for share in shares for j in share) for shares in _worker_mini_batches(shards, 8)]
    expected = get_minibatch_balanced_partitions(
        calculate_workload(torch.tensor(lengths)),
        k_partitions,
        mini_batch_size=8,
        shuffle=shuffle,
        seed=[42, trainer.global_steps],
    )
    for actual_shares, expected_shares in zip(
        _worker_mini_batches(shards, 8), _worker_mini_batches(expected, 8), strict=True
    ):
        assert [sorted(share) for share in actual_shares] == [sorted(share) for share in expected_shares]
    if not shuffle:
        assert mini_batches == [list(range(i * 8, (i + 1) * 8)) for i in range(4)]
    else:
        assert mini_batches != [list(range(i * 8, (i + 1) * 8)) for i in range(4)]
