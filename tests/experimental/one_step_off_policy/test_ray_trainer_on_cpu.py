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

import asyncio

import pytest
from omegaconf import OmegaConf

from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer


async def _none_batch():
    return None


def test_fit_generate_reports_dataloader_exhaustion_on_cpu():
    trainer = OneStepOffRayTrainer.__new__(OneStepOffRayTrainer)
    trainer.metrics = {}
    trainer.timing_raw = {}
    trainer.global_steps = 5
    trainer.total_training_steps = 10
    trainer.config = OmegaConf.create({"data": {"train_batch_size": 8}})

    async def run():
        batch_data_future = asyncio.create_task(_none_batch())
        with pytest.raises(RuntimeError) as exc_info:
            await trainer._fit_generate(batch_data_future, iter(()))
        message = str(exc_info.value)
        assert "Training dataloader was exhausted" in message
        assert "data.train_batch_size=8" in message
        assert "data.train_max_samples=N/A" in message
        assert "trainer.total_epochs=N/A" in message
        assert "trainer.total_training_steps=N/A" in message

    asyncio.run(run())
