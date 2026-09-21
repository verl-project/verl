# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 RainieLLM
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

from pathlib import Path
from unittest.mock import Mock, sentinel

import pytest
from hydra import compose, initialize_config_dir

from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.workers.config import FSDPCriticConfig, HFModelConfig, TrainingWorkerConfig


@pytest.fixture
def trainer(tmp_path):
    # A local model config exercises HFModelConfig without downloading weights or a tokenizer.
    (tmp_path / "config.json").write_text('{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}')
    config_dir = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                f"critic.model.path={tmp_path}",
                "+critic.model.load_tokenizer=false",
                "critic.ppo_micro_batch_size_per_gpu=2",
                "critic.ppo_max_token_len_per_gpu=4096",
                "++critic.ppo_infer_max_token_len_per_gpu=2048",
            ],
        )

    trainer = object.__new__(SeparateRayPPOTrainer)
    trainer.config = config
    trainer.use_critic = True
    trainer.resource_pool_manager = Mock()
    trainer.resource_pool_manager.get_resource_pool.return_value = sentinel.pool
    trainer.resource_pool_to_cls = {sentinel.pool: {}}
    trainer.role_worker_mapping = {Role.Critic: sentinel.worker}
    return trainer


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_create_critic_uses_fsdp_engine_config(trainer, strategy):
    trainer.config.critic.strategy = strategy

    trainer._create_critic_class()

    critic = trainer.orig_critic_cfg
    assert isinstance(critic, FSDPCriticConfig)
    assert isinstance(critic.model, HFModelConfig)
    assert not hasattr(critic.model, "fsdp_config")
    trainer.resource_pool_manager.get_resource_pool.assert_called_once_with(Role.Critic)
    worker = trainer.resource_pool_to_cls[sentinel.pool][str(Role.Critic)]
    worker_config = worker.kwargs["config"]
    assert worker.cls is sentinel.worker
    assert isinstance(worker_config, TrainingWorkerConfig)
    assert worker_config.model_type == "value_model"
    assert worker_config.model_config is critic.model
    assert worker_config.engine_config is critic.engine
    assert worker_config.engine_config is critic.fsdp
    assert worker_config.engine_config.strategy == strategy
    assert worker_config.engine_config.max_token_len_per_gpu == 4096
    assert worker_config.engine_config.infer_max_token_len_per_gpu == 2048
    assert worker_config.optimizer_config is critic.optim
    assert worker_config.checkpoint_config is critic.checkpoint


def test_create_critic_skips_disabled_critic(trainer):
    trainer.use_critic = False

    trainer._create_critic_class()

    trainer.resource_pool_manager.get_resource_pool.assert_not_called()
    assert trainer.resource_pool_to_cls[sentinel.pool] == {}
