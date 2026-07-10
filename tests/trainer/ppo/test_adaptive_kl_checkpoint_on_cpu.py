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

import ast
import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.checkpoint_utils import (
    load_adaptive_kl_controller_state,
    save_adaptive_kl_controller_state,
)
from verl.trainer.ppo.core_algos import AdaptiveKLController

pytestmark = [
    pytest.mark.filterwarnings("ignore:Ray state API is no longer experimental:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:NPU not support router replay for now.:UserWarning"),
]


def _advance(controller, observations):
    for current_kl, n_steps in observations:
        controller.update(current_kl=current_kl, n_steps=n_steps)


def test_resume_matches_uninterrupted_controller(tmp_path):
    observations = [(1.7, 128), (0.2, 64), (1.4, 256), (0.5, 32)]
    uninterrupted = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    _advance(uninterrupted, observations)

    checkpointed = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    _advance(checkpointed, observations[:2])
    save_adaptive_kl_controller_state(checkpointed, tmp_path)

    resumed = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    assert load_adaptive_kl_controller_state(resumed, tmp_path)
    _advance(resumed, observations[2:])

    assert resumed.value == pytest.approx(uninterrupted.value)


def test_missing_state_preserves_initial_value(tmp_path, caplog):
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    assert not load_adaptive_kl_controller_state(controller, tmp_path)
    assert controller.value == 0.1
    assert "No adaptive KL controller state" in caplog.text


def test_malformed_state_fails_closed(tmp_path):
    torch.save({"version": torch.tensor(1), "value": torch.ones(2)}, tmp_path / "kl_ctrl.pt")
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    with pytest.raises(ValueError, match="scalar"):
        load_adaptive_kl_controller_state(controller, tmp_path)


def test_nonfinite_state_fails_closed(tmp_path):
    torch.save({"version": torch.tensor(1), "value": torch.tensor(float("nan"))}, tmp_path / "kl_ctrl.pt")
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    with pytest.raises(ValueError, match="finite"):
        load_adaptive_kl_controller_state(controller, tmp_path)


def test_non_integral_version_fails_closed(tmp_path):
    torch.save({"version": torch.tensor(1.5), "value": torch.tensor(0.2)}, tmp_path / "kl_ctrl.pt")
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    with pytest.raises(ValueError, match="version"):
        load_adaptive_kl_controller_state(controller, tmp_path)


def test_legacy_float_state_remains_loadable(tmp_path):
    torch.save({"value": 0.25}, tmp_path / "kl_ctrl.pt")
    controller = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000)
    assert load_adaptive_kl_controller_state(controller, tmp_path)
    assert controller.value == pytest.approx(0.25)


@pytest.mark.parametrize(
    "relative_path",
    [
        "verl/trainer/ppo/ray_trainer.py",
        "verl/trainer/ppo/v1/trainer_base.py",
        "verl/experimental/fully_async_policy/fully_async_trainer.py",
    ],
)
def test_all_trainer_checkpoint_paths_persist_adaptive_kl_state(relative_path):
    """Keep the legacy, V1, and fully-async save/load paths wired to the helper."""
    source_path = Path(__file__).parents[3] / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    called_helpers = {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "save_adaptive_kl_controller_state" in called_helpers
    assert "load_adaptive_kl_controller_state" in called_helpers


class _FakeDataLoader:
    def __init__(self):
        self.loaded = None

    def __len__(self):
        return 10

    def state_dict(self):
        return {"position": 3}

    def load_state_dict(self, state):
        self.loaded = state


class _FakeWorker:
    def __init__(self):
        self.loaded = None

    def save_checkpoint(self, *args, **kwargs):
        return None

    def load_checkpoint(self, *args, **kwargs):
        self.loaded = (args, kwargs)


def _checkpoint_config(root, resume_path=None):
    return OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(root),
                "default_hdfs_dir": None,
                "resume_mode": "resume_path" if resume_path else "disable",
                "resume_from_path": str(resume_path) if resume_path else None,
                "del_local_ckpt_after_load": False,
                "remove_previous_ckpt_in_save": False,
                "max_actor_ckpt_to_keep": None,
                "max_critic_ckpt_to_keep": None,
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
        }
    )


def _fake_trainer(root, resume_path=None):
    return SimpleNamespace(
        config=_checkpoint_config(root, resume_path),
        actor_rollout_wg=_FakeWorker(),
        critic_wg=_FakeWorker(),
        train_dataloader=_FakeDataLoader(),
        use_critic=False,
        kl_ctrl_in_reward=AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10_000),
        global_steps=3,
    )


def test_legacy_checkpoint_methods_round_trip_controller(tmp_path):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = _fake_trainer(tmp_path)
    trainer.kl_ctrl_in_reward.update(current_kl=2.0, n_steps=128)
    RayPPOTrainer._save_checkpoint(trainer)
    checkpoint = tmp_path / "global_step_3"
    assert (checkpoint / "kl_ctrl.pt").exists()

    resumed = _fake_trainer(tmp_path, checkpoint)
    RayPPOTrainer._load_checkpoint(resumed)
    assert resumed.kl_ctrl_in_reward.value == pytest.approx(trainer.kl_ctrl_in_reward.value)


def _load_v1_trainer_base():
    module_names = ("transfer_queue", "verl.trainer.ppo.v1.trainer_base")
    missing = object()
    previous_modules = {name: sys.modules.get(name, missing) for name in module_names}
    transfer_queue = types.ModuleType("transfer_queue")

    class KVBatchMeta:
        pass

    transfer_queue.KVBatchMeta = KVBatchMeta
    for name in ("init", "kv_clear", "kv_batch_put", "kv_batch_get", "kv_list"):
        setattr(transfer_queue, name, lambda *args, **kwargs: None)
    transfer_queue.async_kv_put = lambda *args, **kwargs: None
    transfer_queue.async_kv_batch_put = lambda *args, **kwargs: None
    sys.modules.setdefault("transfer_queue", transfer_queue)
    try:
        from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    finally:
        for module_name, previous_module in previous_modules.items():
            if previous_module is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module

    return PPOTrainer


def test_v1_checkpoint_methods_round_trip_controller(tmp_path):
    PPOTrainer = _load_v1_trainer_base()
    trainer = _fake_trainer(tmp_path)
    trainer.kl_ctrl_in_reward.update(current_kl=2.0, n_steps=128)
    PPOTrainer._save_checkpoint(trainer)
    checkpoint = tmp_path / "global_step_3"
    assert (checkpoint / "kl_ctrl.pt").exists()

    resumed = _fake_trainer(tmp_path, checkpoint)
    PPOTrainer._load_checkpoint(resumed)
    assert resumed.kl_ctrl_in_reward.value == pytest.approx(trainer.kl_ctrl_in_reward.value)


def test_v1_loader_restores_module_cache(monkeypatch):
    module_names = ("transfer_queue", "verl.trainer.ppo.v1.trainer_base")
    for module_name in module_names:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    _load_v1_trainer_base()

    for module_name in module_names:
        assert module_name not in sys.modules


class _FakeRemoteMethod:
    def remote(self, *args, **kwargs):
        return None


class _FakeRollouter:
    save_checkpoint = _FakeRemoteMethod()


def test_fully_async_checkpoint_methods_round_trip_controller(tmp_path, monkeypatch):
    from verl.experimental.fully_async_policy import fully_async_trainer as fully_async_module

    trainer = _fake_trainer(tmp_path)
    trainer.current_param_version = 3
    trainer.trigger_parameter_sync_step = 1
    trainer.last_ckpt_version = 0
    trainer.rollouter = _FakeRollouter()
    trainer.kl_ctrl_in_reward.update(current_kl=2.0, n_steps=128)
    monkeypatch.setattr(fully_async_module.ray, "get", lambda result: result)
    fully_async_class = fully_async_module.FullyAsyncTrainer.__ray_metadata__.modified_class
    fully_async_class._save_checkpoint(trainer)
    checkpoint = tmp_path / "global_step_3"
    assert (checkpoint / "kl_ctrl.pt").exists()

    resumed = _fake_trainer(tmp_path, checkpoint)
    resumed.current_param_version = 0
    resumed.trigger_parameter_sync_step = 1
    resumed.last_ckpt_version = 0
    value = asyncio.run(fully_async_class.load_checkpoint(resumed))
    assert value == 3
    assert resumed.kl_ctrl_in_reward.value == pytest.approx(trainer.kl_ctrl_in_reward.value)
