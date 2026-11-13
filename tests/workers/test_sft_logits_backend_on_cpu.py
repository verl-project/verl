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

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.sft_trainer import SFTTrainer


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "veomni", "automodel"])
@pytest.mark.parametrize("feature", ["distillation", "teacher_dump"])
def test_logits_consumers_reject_unsupported_backend_before_dataset_loading(monkeypatch, strategy, feature):
    config = {"trainer": {}}
    if feature == "distillation":
        config[feature] = {}
    else:
        config["trainer"][feature] = {"path": "teacher.pt", "topk": 2}

    def build_config(self):
        self.engine_config = SimpleNamespace(strategy=strategy)

    def build_dataset(self):
        pytest.fail("Unsupported logits consumers should fail before loading datasets or models")

    monkeypatch.setattr("verl.trainer.sft_trainer.torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("verl.trainer.sft_trainer.log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(SFTTrainer, "_build_config", build_config)
    monkeypatch.setattr(SFTTrainer, "_build_dataset", build_dataset)
    with pytest.raises(ValueError, match="requires engine.strategy=megatron"):
        SFTTrainer(OmegaConf.create(config))


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "veomni", "automodel", "megatron"])
def test_ce_training_keeps_existing_backend_support(strategy):
    trainer = SimpleNamespace(
        config=OmegaConf.create({"trainer": {}}), engine_config=SimpleNamespace(strategy=strategy)
    )
    SFTTrainer._validate_logits_features(trainer)


def test_megatron_accepts_both_logits_consumers():
    trainer = SimpleNamespace(
        config=OmegaConf.create({"trainer": {"teacher_dump": {}}, "distillation": {}}),
        engine_config=SimpleNamespace(strategy="megatron"),
    )
    SFTTrainer._validate_logits_features(trainer)
