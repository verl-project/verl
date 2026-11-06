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

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from transfer_queue import KVBatchMeta

from verl.protocol import DataProto
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, apply_greedy_sampling_params
from verl.utils.debug.metrics import calculate_debug_metrics


def test_greedy_reward_baseline_does_not_request_score_centering_heads():
    params = {"top_p": 0.9, "top_k": 64, "temperature": 1.0, "topk_log_probs": 128, "logprobs": True}
    apply_greedy_sampling_params(params)
    assert params == {"top_p": 1.0, "top_k": -1, "temperature": 0, "logprobs": True}


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


@pytest.mark.parametrize("score_centering", [True, False])
def test_update_actor_passes_score_centering_flag(score_centering):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 2,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                    "ppo_epochs": 1,
                    "data_loader_seed": 1,
                    "shuffle": False,
                    "policy_loss": {"rollout_correction": {"score_centering": score_centering}},
                },
                "rollout": {"n": 4, "temperature": 1.0},
            }
        }
    )
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.update_actor.return_value = {"metrics": {"mfu": 0.0}}
    batch = SimpleNamespace(extra_info={})

    trainer._update_actor(batch, {})

    assert batch.extra_info["score_centering"] is score_centering


@pytest.mark.parametrize("case", ["mixed", "nonfinite", "all_empty", "single_token"])
@pytest.mark.parametrize("empty_sample_count", [0, 2])
@pytest.mark.parametrize("reverse_updates", [False, True])
def test_step_aggregates_rollout_diagnostics_across_local_updates(case, empty_sample_count, reverse_updates):
    """Exercise real trainer-step aggregation with producer outputs and batch metadata.

    Mock preparation and worker updates to isolate four local updates on CPU.
    Check both update orders, padding-derived weights, and mixed/nonfinite/all-empty
    windows. A single-token update has available tokens but undefined std/correlation.
    Retain trajectory metadata and existing metrics in every case.
    """
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.parameter_sync_step = 4
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 16}})
    local_metrics = []
    batches = []
    for index, sample_count in enumerate([1, empty_sample_count, 3, empty_sample_count]):
        available = case != "all_empty" and index in (0, 2)
        response_mask = torch.zeros((4, 2), dtype=torch.long)
        if available:
            response_mask[:sample_count] = 1
        if case == "single_token" and index == 0:
            response_mask[0, 1] = 0
        rollout_log_probs = torch.tensor([[-1.0, -2.0]]).repeat(4, 1)
        actor_log_probs = rollout_log_probs - (0.5 if index == 0 else 1.0)
        if case == "nonfinite" and index == 2:
            actor_log_probs[0, 0] = float("nan")
        data = DataProto.from_dict(
            {
                "rollout_log_probs": rollout_log_probs,
                "old_log_probs": actor_log_probs,
                "response_mask": response_mask,
                "responses": torch.zeros((4, 2)),
            }
        )
        if case == "single_token" and index == 0:
            with pytest.warns(UserWarning, match="degrees of freedom"):
                local_metrics.append(calculate_debug_metrics(data))
        else:
            local_metrics.append(calculate_debug_metrics(data))
        batches.append(
            KVBatchMeta(
                partition_id="train",
                keys=[f"step_{index}_row_{row}" for row in range(4)],
                tags=[{"is_padding": row >= sample_count, "local_step": index} for row in range(4)],
            )
        )

    available_metrics = (local_metrics[0], local_metrics[2])
    if reverse_updates:
        local_metrics.reverse()
        batches.reverse()
    observed_indices = []
    timing_raw = {}

    def step_once(metrics, timing, sample_batch_size):
        """Return the current local batch and attach its precomputed diagnostics."""
        assert timing is timing_raw
        assert sample_batch_size == 4
        index = trainer.local_trigger_step
        observed_indices.append(index)
        metrics.update(local_metrics[index])
        return batches[index]

    metrics = {"existing_metric": 7.0}
    with (
        patch.object(trainer, "prepare_step", return_value={"timing_s/prepare": 0.5}) as prepare_step,
        patch.object(trainer, "_step_once", side_effect=step_once) as local_step,
    ):
        combined = trainer.step(metrics, timing_raw)

    prepare_step.assert_called_once_with()
    assert local_step.call_count == 4
    assert observed_indices == [0, 1, 2, 3]
    assert combined.partition_id == "train"
    assert combined.keys == [key for batch in batches for key in batch.keys]
    assert combined.tags == [tag for batch in batches for tag in batch.tags]
    assert metrics["existing_metric"] == 7.0
    assert metrics["timing_s/prepare"] == 0.5
    assert metrics["training/rollout_probs_diff_valid"] == (0 if case == "all_empty" else 1)
    expected_mismatches = {"all_empty": 0, "single_token": 7}.get(case, 8)
    assert metrics["training/rollout_logprobs_mismatch_count"] == expected_mismatches
    if case in ("mixed", "single_token"):
        assert metrics["training/rollout_logprobs_diff_mean"] == pytest.approx(0.875)
    for key in (
        "training/rollout_probs_diff_mean",
        "training/rollout_probs_diff_std",
        "training/rollout_actor_probs_pearson_corr",
        "training/rollout_logprobs_diff_mean",
        "training/rollout_probs_diff_max",
        "training/rollout_logprobs_diff_max",
    ):
        if case in ("nonfinite", "all_empty") or (
            case == "single_token"
            and key in ("training/rollout_probs_diff_std", "training/rollout_actor_probs_pearson_corr")
        ):
            assert math.isnan(metrics[key])
        elif key.endswith("_max"):
            assert metrics[key] == pytest.approx(max(available_metrics[0][key], available_metrics[1][key]))
        else:
            # The two available updates have 1 and 3 non-padding samples, despite both returning four rows.
            expected = (available_metrics[0][key] + 3 * available_metrics[1][key]) / 4
            assert metrics[key] == pytest.approx(expected)
