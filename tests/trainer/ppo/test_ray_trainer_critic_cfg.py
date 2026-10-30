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
"""Regression tests for critic config construction in RayPPOTrainer."""

from omegaconf import OmegaConf

from verl.trainer.ppo import ray_trainer


def test_build_critic_cfg_for_worker_skips_driver_dataclass_in_legacy_mode(monkeypatch):
    config = OmegaConf.create({"critic": {"strategy": "fsdp", "model": {"path": "dummy"}}})
    called = {"value": False}

    def _fake_omega_conf_to_dataclass(_):
        called["value"] = True
        return {"converted": True}

    monkeypatch.setattr(ray_trainer, "omega_conf_to_dataclass", _fake_omega_conf_to_dataclass)

    critic_cfg = ray_trainer.build_critic_cfg_for_worker(config, use_legacy_worker_impl="auto")

    assert critic_cfg is config.critic
    assert called["value"] is False


def test_build_critic_cfg_for_worker_keeps_dataclass_path_in_new_worker_mode(monkeypatch):
    config = OmegaConf.create({"critic": {"strategy": "fsdp", "model": {"path": "dummy"}}})

    def _fake_omega_conf_to_dataclass(_):
        return {"converted": True}

    monkeypatch.setattr(ray_trainer, "omega_conf_to_dataclass", _fake_omega_conf_to_dataclass)

    critic_cfg = ray_trainer.build_critic_cfg_for_worker(config, use_legacy_worker_impl="disable")

    assert critic_cfg == {"converted": True}
