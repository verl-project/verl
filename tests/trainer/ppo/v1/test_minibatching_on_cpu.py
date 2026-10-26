# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.utils import (
    get_minibatch_padding_multiple,
    validate_minibatching_config,
)
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _minibatching_config(*, dynamic=True, shuffle=False, epochs=1):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 4,
                    "ppo_micro_batch_size_per_gpu": None if dynamic else 2,
                    "use_dynamic_bsz": dynamic,
                    "shuffle": shuffle,
                    "ppo_epochs": epochs,
                    "data_loader_seed": 42,
                },
                "rollout": {
                    "n": 2,
                    "log_prob_use_dynamic_bsz": dynamic,
                    "log_prob_micro_batch_size_per_gpu": None if dynamic else 3,
                },
                "ref": {
                    "log_prob_use_dynamic_bsz": dynamic,
                    "log_prob_micro_batch_size_per_gpu": None if dynamic else 4,
                },
            },
            "critic": {
                "ppo_mini_batch_size": 2,
                "ppo_micro_batch_size_per_gpu": None if dynamic else 5,
                "ppo_infer_micro_batch_size_per_gpu": None,
                "forward_micro_batch_size_per_gpu": None if dynamic else 6,
                "use_dynamic_bsz": dynamic,
                "shuffle": False,
                "ppo_epochs": 1,
            },
            "trainer": {"critic_warmup": 0},
        }
    )


def test_padding_multiple_covers_all_static_micro_batch_sizes():
    config = _minibatching_config(dynamic=False)

    assert (
        get_minibatch_padding_multiple(
            config,
            dp_size=2,
            use_critic=True,
            use_reference_policy=True,
            update_actor=True,
        )
        == 120
    )


def test_dynamic_padding_only_requires_dp_divisibility():
    config = _minibatching_config(dynamic=True)

    assert (
        get_minibatch_padding_multiple(
            config,
            dp_size=4,
            use_critic=True,
            use_reference_policy=True,
            update_actor=True,
        )
        == 4
    )


def test_multi_epoch_shuffle_is_rejected():
    config = _minibatching_config(shuffle=True, epochs=2)

    with pytest.raises(ValueError, match="does not support actor.shuffle=true"):
        validate_minibatching_config(config, use_critic=False)


def test_balance_batch_allows_partial_final_uid_minibatches(monkeypatch):
    config = _minibatching_config(dynamic=True)
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = config
    trainer.use_critic = True
    trainer.use_reference_policy = False
    trainer.global_steps = 1
    trainer.local_trigger_step = 0
    trainer.tokenizer = SimpleNamespace(eos_token_id=0)
    trainer.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": [0, 1]})

    output_counts = {"p0": 2, "p1": 1, "p2": 3, "p3": 2, "p4": 1, "p5": 4, "p6": 2}
    uid_keys = {
        uid: [f"{uid}_0_{output_idx}" for output_idx in range(output_count)]
        for uid, output_count in output_counts.items()
    }
    keys = [key for uid in output_counts for key in uid_keys[uid]]
    batch = KVBatchMeta(
        partition_id="test",
        keys=keys,
        tags=[{"seq_len": index + 1} for index in range(len(keys))],
        extra_info={},
    )

    def fake_append_padding_samples(batch, pad_size, eos_token_id):
        del eos_token_id
        pad_keys = [f"pad{i}_0_0" for i in range(pad_size)]
        return KVBatchMeta(
            partition_id=batch.partition_id,
            keys=batch.keys + pad_keys,
            tags=batch.tags + [{"seq_len": 128, "is_padding": True} for _ in pad_keys],
            extra_info=batch.extra_info,
        )

    monkeypatch.setattr(
        "verl.trainer.ppo.v1.trainer_base.append_padding_samples",
        fake_append_padding_samples,
    )

    result = trainer._balance_batch(batch, metrics={})

    layout = result.extra_info["_minibatch_layout"]
    assert layout == {
        "unit_prompt_count": 2,
        "unit_global_sizes": [4, 6, 6, 2],
        "unit_padding_counts": [1, 1, 1, 0],
    }
    assert len(result) == 18

    def reconstruct_global_minibatches(global_sizes):
        shards = [result.keys[: len(result) // 2], result.keys[len(result) // 2 :]]
        local_offset = 0
        mini_batches = []
        for global_size in global_sizes:
            local_size = global_size // len(shards)
            mini_batches.append([key for shard in shards for key in shard[local_offset : local_offset + local_size]])
            local_offset += local_size
        assert all(local_offset == len(shard) for shard in shards)
        return mini_batches

    def assert_uid_minibatches(prompt_mini_batch_size, expected_global_sizes, expected_padding_counts):
        layout = trainer._get_global_mini_batch_layout(result, prompt_mini_batch_size)
        assert layout == {
            "global_mini_batch_sizes": expected_global_sizes,
            "global_mini_batch_padding_counts": expected_padding_counts,
        }
        mini_batches = reconstruct_global_minibatches(expected_global_sizes)
        uids = list(output_counts)
        for index, mini_batch in enumerate(mini_batches):
            expected_uids = uids[index * prompt_mini_batch_size : (index + 1) * prompt_mini_batch_size]
            expected_keys = [key for uid in expected_uids for key in uid_keys[uid]]
            real_keys = [key for key in mini_batch if not key.startswith("pad")]
            assert sorted(real_keys) == sorted(expected_keys)

    assert_uid_minibatches(config.actor_rollout_ref.actor.ppo_mini_batch_size, [10, 8], [2, 1])
    assert_uid_minibatches(config.critic.ppo_mini_batch_size, [4, 6, 6, 2], [1, 1, 1, 0])


def test_step_restores_minibatch_layout_after_metadata_rebuild():
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = _minibatching_config(dynamic=True)
    trainer.config.actor_rollout_ref.rollout.temperature = 1.0
    trainer.global_steps = 1
    trainer.use_critic = False
    trainer.use_reference_policy = False
    trainer.actor_step_residency = False
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())

    layout = {
        "unit_prompt_count": 2,
        "unit_global_sizes": [4],
        "unit_padding_counts": [0],
    }
    source_batch = KVBatchMeta(
        partition_id="test",
        keys=["p0_0_0", "p0_0_1", "p1_0_0", "p1_0_1"],
        tags=[{"seq_len": 8}] * 4,
        extra_info={},
    )

    trainer.replay_buffer = SimpleNamespace(sample=lambda **kwargs: (source_batch, {}))

    def balance_batch(batch, metrics):
        batch.extra_info["_minibatch_layout"] = layout
        return batch

    trainer._balance_batch = balance_batch

    def rebuild_metadata(batch, metrics):
        return KVBatchMeta(
            partition_id=batch.partition_id,
            keys=list(batch.keys),
            tags=list(batch.tags),
            extra_info={},
        )

    trainer._compute_old_log_prob = rebuild_metadata
    trainer._compute_advantage = rebuild_metadata

    def update_actor(batch, metrics):
        assert batch.extra_info["_minibatch_layout"] == layout
        return batch

    trainer._update_actor = update_actor

    result = trainer._step_once(metrics={}, timing_raw={}, sample_batch_size=2)

    assert result.extra_info["_minibatch_layout"] == layout
