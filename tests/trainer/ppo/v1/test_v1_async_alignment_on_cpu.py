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
"""CPU tests for the v1 async-trainer no-hybrid-replicas path and off-policy bypass.

Covers the ``hybrid_rollout.enable_switch=False`` no-op mode-switch hooks and the
``max_off_policy_threshold=None`` off-policy control bypass.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.streaming_dataloader import StreamingDataloaderAsync
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync


def _make_dataloader(*, max_off_policy_threshold, max_off_policy_strategy) -> StreamingDataloaderAsync:
    dataloader = object.__new__(StreamingDataloaderAsync)
    dataloader._init_replay_state(
        max_off_policy_threshold=max_off_policy_threshold,
        max_off_policy_strategy=max_off_policy_strategy,
        sampler_kwargs={},
        poll_interval=0.05,
    )
    return dataloader


def _make_trainer(
    *,
    standalone_replicas=1,
    hybrid_replicas=2,
    current_mode=HybridEngineMode.ROLLOUT,
    enable_hybrid_replicas=True,
) -> PPOTrainerSeparateAsync:
    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer._enable_hybrid_replicas = enable_hybrid_replicas
    trainer.current_mode = current_mode
    trainer.llm_server_manager = SimpleNamespace(
        get_replicas=lambda: ["hybrid"] * hybrid_replicas,
        server_addresses=[f"hybrid-{i}" for i in range(hybrid_replicas)],
    )
    trainer.standalone_server_manager = SimpleNamespace(
        get_replicas=lambda: ["standalone"] * standalone_replicas,
        global_load_balancer=MagicMock(),
    )
    trainer.checkpoint_manager = MagicMock()
    return trainer


class TestHybridReplicasDisabled:
    """When hybrid_rollout.enable_switch=False, mode-switch hooks must be no-ops."""

    def test_resource_mapping_uses_pure_actor_role(self):
        trainer = object.__new__(PPOTrainerSeparateAsync)
        trainer._enable_hybrid_replicas = False
        trainer.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"lora": {"rank": 0}, "lora_adapter_path": None},
                    "actor": {"use_kl_loss": False},
                },
                "algorithm": {},
                "critic": {"enable": False},
                "trainer": {"nnodes": 1, "n_gpus_per_node": 4},
                "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
                "distillation": None,
            }
        )

        trainer._init_resource_pool_mgr()

        assert Role.Actor in trainer.role_worker_mapping
        assert Role.ActorRollout not in trainer.role_worker_mapping
        assert Role.ActorRolloutRef not in trainer.role_worker_mapping

    def test_mode_switch_hooks_are_noops(self):
        trainer = _make_trainer(enable_hybrid_replicas=False)
        trainer.switch_to_rollout()
        trainer.switch_to_trainer()
        trainer.add_replicas_to_balancer()
        trainer.remove_replicas_from_balancer()
        trainer.checkpoint_manager.update_weights.assert_not_called()
        trainer.checkpoint_manager.abort_replicas.assert_not_called()
        trainer.checkpoint_manager.sleep_replicas.assert_not_called()

    def test_on_init_end_skips_hybrid_weight_update(self):
        trainer = _make_trainer(enable_hybrid_replicas=False)
        trainer.global_steps = 1
        trainer.standalone_checkpoint_manager = MagicMock()
        trainer.on_init_end()
        trainer.standalone_checkpoint_manager.update_weights.assert_called_once()
        trainer.checkpoint_manager.update_weights.assert_not_called()


class TestOffPolicyThresholdNone:
    def test_none_threshold_disables_stale_eviction(self):
        rb = _make_dataloader(max_off_policy_threshold=None, max_off_policy_strategy="drop")
        # A very old terminal key stays sampleable: no drop even with strategy="drop".
        rb.prompt_global_steps["train"]["stale"] = 1
        rb.finished_keys["train"].add("stale")
        assert rb._stale_terminal_keys(1000, "train") == set()

    def test_none_threshold_disables_wait_blocking(self):
        rb = _make_dataloader(max_off_policy_threshold=None, max_off_policy_strategy="wait")
        rb.prompt_global_steps["train"]["old"] = 1
        rb.pending_keys["train"].add("old")
        # Enough sampleable keys exist: None threshold must not block on the stale pending key.
        rb.finished_keys["train"].update({f"k{i}" for i in range(4)})
        assert rb._has_enough_samples(1000, "train", 4, set(rb.finished_keys["train"])) is True
