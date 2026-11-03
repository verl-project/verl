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

from verl.trainer.ppo.padding_utils import PADDING_MASK_KEY
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


def _fake_append_padding_samples(batch, pad_size, eos_token_id):
    del eos_token_id
    pad_keys = [f"pad{i}_0_0" for i in range(pad_size)]
    return KVBatchMeta(
        partition_id=batch.partition_id,
        keys=batch.keys + pad_keys,
        tags=batch.tags + [{"seq_len": 128, "is_padding": True} for _ in pad_keys],
        extra_info=batch.extra_info,
    )


@pytest.fixture
def stub_padding(monkeypatch):
    monkeypatch.setattr(
        "verl.trainer.ppo.v1.trainer_base.append_padding_samples",
        _fake_append_padding_samples,
    )


def _minibatching_config(*, mode="fixed_count", dynamic=True, shuffle=False, epochs=1):
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
            "trainer": {"critic_warmup": 0, "v1": {"mini_batch_mode": mode}},
        }
    )


def _trainer(config, *, use_critic=True):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = config
    trainer.use_critic = use_critic
    trainer.use_reference_policy = False
    trainer.global_steps = 1
    trainer.local_trigger_step = 0
    trainer.tokenizer = SimpleNamespace(eos_token_id=0)
    trainer.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": [0, 1]})
    return trainer


def _batch(keys):
    return KVBatchMeta(
        partition_id="test",
        keys=keys,
        tags=[{"seq_len": index + 1} for index in range(len(keys))],
        extra_info={},
    )


def _global_minibatches(batch, global_sizes, dp_size=2):
    local_batch_size = len(batch) // dp_size
    shards = [batch.keys[rank * local_batch_size : (rank + 1) * local_batch_size] for rank in range(dp_size)]
    offsets = [0]
    for size in global_sizes:
        offsets.append(offsets[-1] + size // dp_size)
    return [
        [key for shard in shards for key in shard[start:end]]
        for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    ]


@pytest.mark.parametrize(("dynamic", "dp_size", "expected"), [(False, 2, 120), (True, 4, 4)])
def test_padding_multiple_satisfies_active_stages(dynamic, dp_size, expected):
    assert (
        get_minibatch_padding_multiple(
            _minibatching_config(dynamic=dynamic),
            dp_size=dp_size,
            use_critic=True,
            use_reference_policy=True,
            update_actor=True,
        )
        == expected
    )


def test_fixed_count_multi_epoch_shuffle_is_rejected():
    config = _minibatching_config(shuffle=True, epochs=2)

    with pytest.raises(ValueError, match="does not support actor.shuffle=true"):
        validate_minibatching_config(config, use_critic=False)


def test_fixed_rows_multi_epoch_shuffle_is_allowed():
    config = _minibatching_config(mode="fixed_rows", shuffle=True, epochs=2)

    validate_minibatching_config(config, use_critic=True)


def test_missing_mode_defaults_to_fixed_rows():
    config = _minibatching_config()
    del config.trainer.v1.mini_batch_mode
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = config

    assert trainer._get_mini_batch_mode() == "fixed_rows"


def test_invalid_mini_batch_mode_is_rejected():
    config = _minibatching_config(mode="unknown")

    with pytest.raises(ValueError, match="trainer.v1.mini_batch_mode"):
        validate_minibatching_config(config, use_critic=False)


def test_fixed_rows_builds_fixed_trajectory_minibatches(stub_padding):
    config = _minibatching_config(mode="fixed_rows", dynamic=True)
    trainer = _trainer(config)
    result = trainer._balance_batch(_batch([f"row{index}" for index in range(15)]), metrics={})

    assert result.extra_info["_minibatch_layout"] == {
        "unit_count": 4,
        "unit_global_sizes": [4, 4, 4, 4],
        "unit_padding_counts": [0, 0, 0, 1],
    }
    assert trainer._get_global_mini_batch_layout(result, 8) == {
        "global_mini_batch_sizes": [8, 8],
        "global_mini_batch_padding_counts": [0, 1],
    }
    assert trainer._get_global_mini_batch_layout(result, 4) == {
        "global_mini_batch_sizes": [4, 4, 4, 4],
        "global_mini_batch_padding_counts": [0, 0, 0, 1],
    }


def test_fixed_rows_multi_epoch_shuffle_uses_whole_batch_fallback(stub_padding):
    config = _minibatching_config(mode="fixed_rows", dynamic=True, shuffle=True, epochs=2)
    trainer = _trainer(config, use_critic=False)
    result = trainer._balance_batch(_batch([f"row{index}" for index in range(5)]), metrics={})
    batching_info = trainer._get_update_minibatching_info(result, config.actor_rollout_ref.actor)

    assert "_minibatch_layout" not in result.extra_info
    assert batching_info == {
        "global_batch_size": 8,
        "mini_batch_size": 8,
        "dataloader_kwargs": {"shuffle": True},
        PADDING_MASK_KEY: [bool(tag.get("is_padding", False)) for tag in result.tags],
    }


def test_fixed_count_preserves_multi_output_uid_minibatches(stub_padding):
    config = _minibatching_config(dynamic=True)
    trainer = _trainer(config)
    output_counts = {"p0": 2, "p1": 1, "p2": 3, "p3": 2, "p4": 1}
    keys = [f"{uid}_0_{output_index}" for uid, count in output_counts.items() for output_index in range(count)]
    result = trainer._balance_batch(_batch(keys), metrics={})

    assert result.extra_info["_minibatch_layout"] == {
        "unit_count": 2,
        "unit_global_sizes": [4, 6, 2],
        "unit_padding_counts": [1, 1, 1],
    }
    actor_layout = trainer._get_global_mini_batch_layout(result, 4)
    critic_layout = trainer._get_global_mini_batch_layout(result, 2)
    assert actor_layout == {
        "global_mini_batch_sizes": [10, 2],
        "global_mini_batch_padding_counts": [2, 1],
    }
    assert critic_layout == {
        "global_mini_batch_sizes": [4, 6, 2],
        "global_mini_batch_padding_counts": [1, 1, 1],
    }

    def uid_sets(global_sizes):
        return [
            {key.rsplit("_", 2)[0] for key in mini_batch if not key.startswith("pad")}
            for mini_batch in _global_minibatches(result, global_sizes)
        ]

    assert uid_sets(actor_layout["global_mini_batch_sizes"]) == [{"p0", "p1", "p2", "p3"}, {"p4"}]
    assert uid_sets(critic_layout["global_mini_batch_sizes"]) == [{"p0", "p1"}, {"p2", "p3"}, {"p4"}]


def test_step_restores_minibatch_layout_after_metadata_rebuild():
    trainer = _trainer(_minibatching_config(dynamic=True), use_critic=False)
    trainer.config.actor_rollout_ref.rollout.temperature = 1.0
    trainer.actor_step_residency = False
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())

    layout = {"unit_count": 2, "unit_global_sizes": [4], "unit_padding_counts": [0]}
    source_batch = _batch(["p0_0_0", "p0_0_1", "p1_0_0", "p1_0_1"])

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
    trainer._update_actor = lambda batch, metrics: batch

    result = trainer._step_once(metrics={}, timing_raw={}, sample_batch_size=2)

    assert result.extra_info["_minibatch_layout"] == layout
