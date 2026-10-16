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

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from omegaconf import OmegaConf

from verl.utils import tensordict_utils as tu


@pytest.fixture(params=["sft_trainer_ray", "sft_trainer"], ids=["ray", "spmd"])
def trainer_setup(request, monkeypatch):
    module = importlib.import_module(f"verl.trainer.{request.param}")
    tracking = Mock()
    monkeypatch.setattr(module, "Tracking", tracking)
    monkeypatch.setattr(module, "log_with_rank", Mock())
    if request.param == "sft_trainer":
        monkeypatch.setattr(module, "aggressive_empty_cache", Mock())
        monkeypatch.setattr(module, "log_gpu_memory_usage", Mock())

    trainer = module.SFTTrainer.__new__(module.SFTTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "resume",
                "logger": [],
                "total_epochs": 2,
                "balance_batch": False,
            },
            "model": {"use_remove_padding": False},
            "data": {
                "use_dynamic_bsz": False,
                "temperature": 1.0,
                "max_token_len_per_gpu": 8,
                "micro_batch_size_per_gpu": 1,
                "pad_mode": "right",
            },
        }
    )
    trainer.steps_per_epoch = 4
    trainer.global_batch_size = 1
    trainer.model_config = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    trainer.device_name = "cpu"
    trainer.rank = 0
    trainer.start_profile_step = -1
    trainer.end_profile_step = -1
    trainer.test_freq = -1
    trainer.save_freq = -1
    trainer.val_dataloader = None
    trainer.train_sampler = Mock()
    trainer.ckpt_handler = Mock()
    trainer.engine = Mock()
    trainer.engine.is_mp_src_rank_with_outputs.return_value = True
    trainer.engine.get_data_parallel_rank.return_value = 0

    batches_read = []

    def batches():
        for index in range(trainer.steps_per_epoch):
            batches_read.append(index)
            yield {"input_ids": torch.ones((1, 1), dtype=torch.long)}

    trainer.train_dataloader = MagicMock()
    trainer.train_dataloader.__iter__.side_effect = batches
    trainer.train_dataloader.__len__.return_value = trainer.steps_per_epoch
    trainer._get_batch_seqlens = Mock(return_value=torch.tensor([1]) if request.param == "sft_trainer_ray" else [1])
    output = tu.get_tensordict(
        {"placeholder": torch.ones(1)},
        {"metrics": {"loss": 0.0, "grad_norm": 0.0, "lr": 0.0, "mfu": 0.0}},
    )
    trainer.training_client = Mock()
    trainer.training_client.train_batch.return_value = (
        Mock(get=Mock(return_value=output)) if request.param == "sft_trainer_ray" else output
    )
    return trainer, tracking, batches_read


@pytest.mark.parametrize("resume_step,total_steps", [(4, 4), (5, 4), (0, 0)])
def test_completed_sft_resume_has_no_training_side_effects(trainer_setup, resume_step, total_steps):
    trainer, tracking, batches_read = trainer_setup
    trainer.resume_global_step = resume_step
    trainer.total_training_steps = total_steps

    trainer.fit()

    trainer.training_client.train_batch.assert_not_called()
    trainer.training_client.infer_batch.assert_not_called()
    trainer.ckpt_handler.save_checkpoint.assert_not_called()
    trainer.train_sampler.set_epoch.assert_not_called()
    trainer.train_dataloader.__iter__.assert_not_called()
    tracking.assert_not_called()
    assert batches_read == []


@pytest.mark.parametrize("resume_step,total_steps", [(0, 1), (3, 4), (4, 5)])
def test_unfinished_sft_resume_executes_remaining_step(trainer_setup, resume_step, total_steps):
    trainer, tracking, batches_read = trainer_setup
    trainer.resume_global_step = resume_step
    trainer.total_training_steps = total_steps

    trainer.fit()

    trainer.training_client.train_batch.assert_called_once()
    trainer.ckpt_handler.save_checkpoint.assert_called_once_with(step=total_steps)
    trainer.train_sampler.set_epoch.assert_called_once_with(epoch=resume_step // trainer.steps_per_epoch)
    tracking.assert_called_once()
    assert len(batches_read) == 1
