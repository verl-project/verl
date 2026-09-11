# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import datetime
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _CustomSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )


def _dumped_rollout(keys: list[str], reward_extra_infos: list[dict | None]) -> dict:
    """Run ``_log_rollout_data`` over a stubbed TransferQueue read and return the kwargs it passed
    to ``_dump_generations``. The stub honours ``select_fields`` like the real queue, so a field the
    trainer stops requesting is a field it stops receiving.
    """
    n = len(keys)
    data = {
        "uid": np.array(keys, dtype=object),
        "prompts": torch.nested.nested_tensor([[10 + i] for i in range(n)], layout=torch.jagged),
        "responses": torch.nested.nested_tensor([[20 + i] for i in range(n)], layout=torch.jagged),
        "rm_scores": torch.arange(n, dtype=torch.float32).unsqueeze(1),
        "reward_model": np.array([{"ground_truth": f"gt{i}"} for i in range(n)], dtype=object),
        "extra_fields": np.array(
            [{} if extra is None else {"reward_extra_info": extra} for extra in reward_extra_infos], dtype=object
        ),
    }

    def kv_batch_get(keys, partition_id, select_fields):
        return {field: data[field] for field in select_fields}

    trainer = _StubTrainer.__new__(_StubTrainer)
    # Decode ids to their digits so dumped text stays traceable to its sample.
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, skip_special_tokens=True: str(int(ids[0])))
    batch = KVBatchMeta(keys=list(keys), tags=[{} for _ in keys], partition_id="train")

    with (
        patch("verl.trainer.ppo.v1.trainer_base.tq.kv_batch_get", side_effect=kv_batch_get),
        patch.object(_StubTrainer, "_dump_generations") as dump_generations,
    ):
        trainer._log_rollout_data(batch, {}, "/dev/null/never-written")

    return dump_generations.call_args.kwargs


def test_rollout_dump_carries_reward_extra_info_sorted_by_uid():
    dumped = _dumped_rollout(["u1_0_0", "u0_0_0"], [{"acc": 1.0}, {"acc": 0.0}])

    assert dumped["reward_extra_infos_dict"] == {"acc": [0.0, 1.0], "uid": ["u0_0_0", "u1_0_0"]}
    assert dumped["gts"] == ["gt1", "gt0"]
    assert dumped["scores"] == [1.0, 0.0]
    assert dumped["outputs"] == ["21", "20"]


def test_rollout_dump_pads_reward_extra_info_keys_missing_from_some_samples():
    dumped = _dumped_rollout(["u0_0_0", "u1_0_0", "u2_0_0"], [{"acc": 1.0}, None, {"acc": 0.0, "pred": "x"}])

    assert dumped["reward_extra_infos_dict"] == {
        "acc": [1.0, None, 0.0],
        "pred": [None, None, "x"],
        "uid": ["u0_0_0", "u1_0_0", "u2_0_0"],
    }


def test_generation_dump_writes_sparse_and_exotic_values(tmp_path):
    PPOTrainer._write_generations(
        inputs=["p"],
        outputs=["o"],
        gts=["g"],
        scores=[1.0],
        reward_extra_infos_dict={"acc": [None], "stamp": [datetime.date(2026, 8, 27)]},
        dump_path=str(tmp_path),
        global_steps=3,
    )

    row = json.loads((tmp_path / "3.jsonl").read_text())
    assert row["acc"] is None
    assert row["stamp"] == "2026-08-27"
