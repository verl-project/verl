# Copyright 2026 Amazon.com Inc and/or its affiliates
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

"""The FSDP and Megatron checkpoint managers save the model's generation_config.json with the HF config, also
one that transformers refuses to save, and log a warning when they save none.
"""

import json
import logging
import os

import pytest
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, Qwen3Config

from verl.trainer.config import CheckpointConfig
from verl.utils import transformers_compat
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.megatron_utils import get_hf_model_checkpoint_path

# zai-org/GLM-5.3-Flash's published generation_config.json. It sets top_p without do_sample, which
# GenerationConfig.save_pretrained refuses, except in transformers releases with huggingface/transformers#49914
# (they save a loaded config that is unchanged).
REFUSED_GENERATION_CONFIG = {
    "_from_model_config": True,
    "eos_token_id": [154820, 154827, 154829],
    "pad_token_id": 154820,
    "temperature": 1.0,
    "top_p": 0.95,
    "transformers_version": "5.16.0",
}
SAVED_GENERATION_CONFIG = {"do_sample": True, "eos_token_id": [1, 2], "temperature": 0.6, "top_k": 20, "top_p": 0.95}
# The managers save it with transformers_compat.copy_generation_config, which logs through its module's
# logger, logging.getLogger(__file__).
_LOGGER = transformers_compat.__file__


def _tiny_config():
    return Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
        vocab_size=256,
    )


def _model_dir(tmp_path, generation_config):
    """A model directory with this generation_config.json: a dict, raw text, or None for no file."""
    path = tmp_path / "model"
    path.mkdir()
    if generation_config is not None:
        text = generation_config if isinstance(generation_config, str) else json.dumps(generation_config)
        (path / "generation_config.json").write_text(text, encoding="utf-8")
    return str(path)


def _save_with_fsdp(monkeypatch, local_path, model_path):
    """FSDPCheckpointManager.save_checkpoint of a tiny model whose name_or_path is model_path; returns the HF dir."""
    # A plain module has fsdp_version 0, so the save runs no collectives; only these calls need a stub.
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    model = AutoModelForCausalLM.from_config(_tiny_config())
    model.config.name_or_path = model_path
    manager = FSDPCheckpointManager(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        processing_class=None,
        checkpoint_config=CheckpointConfig(save_contents=["model"], load_contents=["model"], async_save=False),
    )
    manager.save_checkpoint(local_path=local_path, global_step=0)
    return os.path.join(local_path, "huggingface")


def _save_with_megatron(monkeypatch, local_path, model_path):
    """MegatronCheckpointManager's rank-0 HF config save, hf_config.name_or_path = model_path; returns the HF dir."""
    manager = MegatronCheckpointManager.__new__(MegatronCheckpointManager)  # only what this save reads
    manager.rank = 0
    manager.processing_class = None
    manager.hf_config = _tiny_config()
    manager.hf_config.name_or_path = model_path
    manager._save_hf_config_and_tokenizer(local_path)
    return get_hf_model_checkpoint_path(local_path)


SAVES = pytest.mark.parametrize("save", [_save_with_fsdp, _save_with_megatron], ids=["fsdp", "megatron"])


def _records(caplog):
    return [r for r in caplog.records if r.name == _LOGGER]


def _save_pretrained_refuses(model_dir, save_dir):
    """Whether the installed transformers' GenerationConfig.save_pretrained refuses the config in model_dir."""
    try:
        GenerationConfig.from_pretrained(model_dir).save_pretrained(save_dir)
    except ValueError:
        return True
    return False


@SAVES
def test_refused_generation_config_is_saved_with_its_keys_and_values(save, monkeypatch, tmp_path, caplog):
    model_dir = _model_dir(tmp_path, REFUSED_GENERATION_CONFIG)
    refused = _save_pretrained_refuses(model_dir, tmp_path / "save_pretrained")
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        hf_dir = save(monkeypatch, str(tmp_path / "ckpt"), model_dir)

    with open(os.path.join(hf_dir, "generation_config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    expected = dict(REFUSED_GENERATION_CONFIG)
    del saved["transformers_version"], expected["transformers_version"]
    assert saved == expected
    # One info line with transformers' reason when save_pretrained refuses the config. Releases with
    # huggingface/transformers#49914 save it, and then nothing is logged; the fallback stays until verl requires one.
    records = _records(caplog)
    if refused:
        assert [r.levelno for r in records] == [logging.INFO]
        assert hf_dir in records[0].getMessage() and "top_p" in records[0].getMessage()
    else:
        assert not records


@SAVES
def test_generation_config_that_transformers_saves_is_unchanged(save, monkeypatch, tmp_path, caplog):
    model_dir = _model_dir(tmp_path, SAVED_GENERATION_CONFIG)
    expected_dir = tmp_path / "expected"
    GenerationConfig.from_pretrained(model_dir).save_pretrained(expected_dir)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        hf_dir = save(monkeypatch, str(tmp_path / "ckpt"), model_dir)

    with open(os.path.join(hf_dir, "generation_config.json"), "rb") as f:
        saved = f.read()
    assert saved == (expected_dir / "generation_config.json").read_bytes()
    assert not _records(caplog)


@SAVES
@pytest.mark.parametrize("generation_config", [None, "{not json"], ids=["missing", "unreadable"])
def test_generation_config_that_fails_is_logged_and_the_save_goes_on(
    save, generation_config, monkeypatch, tmp_path, caplog
):
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        hf_dir = save(monkeypatch, str(tmp_path / "ckpt"), _model_dir(tmp_path, generation_config))

    assert not os.path.exists(os.path.join(hf_dir, "generation_config.json"))
    assert os.path.isfile(os.path.join(hf_dir, "config.json"))
    records = _records(caplog)
    assert [r.levelno for r in records] == [logging.WARNING]
    assert records[0].getMessage().startswith(f"Saved no generation_config.json to {hf_dir}: ")
