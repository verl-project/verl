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

from verl.workers.config.critic import CriticConfig
from verl.workers.config.engine import EngineConfig
from verl.workers.config.optimizer import OptimizerConfig


def _make_critic(**kwargs) -> CriticConfig:
    defaults = dict(
        strategy="fsdp2",
        use_dynamic_bsz=True,
        ppo_micro_batch_size_per_gpu=2,
        optim=OptimizerConfig(lr=1e-5),
    )
    defaults.update(kwargs)
    return CriticConfig(**defaults)


def test_apply_engine_batching_copies_static_and_dynamic_knobs():
    critic = _make_critic(
        use_dynamic_bsz=False,
        ppo_micro_batch_size_per_gpu=4,
        ppo_infer_micro_batch_size_per_gpu=2,
        ppo_max_token_len_per_gpu=4096,
        ppo_infer_max_token_len_per_gpu=2048,
        forward_max_token_len_per_gpu=1024,
    )
    engine = EngineConfig()
    critic.apply_engine_batching(engine)

    assert engine.use_dynamic_bsz is False
    assert engine.micro_batch_size_per_gpu == 4
    assert engine.infer_micro_batch_size_per_gpu == 2
    assert engine.max_token_len_per_gpu == 4096
    assert engine.infer_max_token_len_per_gpu == 2048


def test_apply_engine_batching_falls_back_infer_micro_batch_to_training():
    critic = _make_critic(
        use_dynamic_bsz=False,
        ppo_micro_batch_size_per_gpu=8,
        ppo_infer_micro_batch_size_per_gpu=None,
    )
    engine = EngineConfig()
    critic.apply_engine_batching(engine)
    assert engine.infer_micro_batch_size_per_gpu == 8


def test_v1_train_budget_is_not_overwritten_by_infer_budget():
    """V1 previously assigned max_token_len_per_gpu = ppo_infer_max_token_len_per_gpu."""
    critic = _make_critic(
        ppo_max_token_len_per_gpu=8192,
        ppo_infer_max_token_len_per_gpu=1024,
    )
    engine = EngineConfig()
    critic.apply_engine_batching(engine)
    assert engine.max_token_len_per_gpu == 8192
    assert engine.infer_max_token_len_per_gpu == 1024
