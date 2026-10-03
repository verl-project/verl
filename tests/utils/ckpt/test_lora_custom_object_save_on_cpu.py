# Copyright 2025 Individual Contributor: Zhiliang Wu
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

"""For a PEFT model (LoRA here) whose config carries ``auto_map`` (a trust_remote_code model),
``FSDPCheckpointManager.save_checkpoint`` bundles the model's own defining source next to the
checkpoint via transformers' ``custom_object_save`` (which copies the source of
``type(obj).__module__``). It must receive the base model, not the ``PeftModel`` wrapper -- this test
pins that the object passed is ``model.get_base_model()``.
"""

from unittest.mock import MagicMock

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen3Config

from verl.trainer.config import CheckpointConfig
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


def _make_peft_model():
    config = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
        vocab_size=256,
    )
    base = AutoModelForCausalLM.from_config(config)
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"
    )
    return get_peft_model(base, lora_config)


def test_custom_object_save_receives_base_model_not_peft_wrapper(monkeypatch, tmp_path):
    """On a LoRA save, the object passed to custom_object_save is the base model, not the PeftModel."""
    # Neutralize the (uninitialized) process group: save_checkpoint queries rank/world_size and
    # calls barrier(). A plain PeftModel has fsdp_version == 0, so the sharded-state-dict block is a
    # nullcontext and no collectives run -- only these three calls need stubbing.
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    peft_model = _make_peft_model()
    # Make the auto_map gate fire (dict without a None key, matching a trust_remote_code config).
    peft_model.config.auto_map = {"AutoModelForCausalLM": "modeling_x.Foo"}

    # save_contents=["model"] keeps the save cheap (skips optimizer/extra) and skips hf_model export.
    ckpt_config = CheckpointConfig(save_contents=["model"], load_contents=["model"], async_save=False)
    manager = FSDPCheckpointManager(
        model=peft_model,
        optimizer=None,
        lr_scheduler=None,
        processing_class=None,
        checkpoint_config=ckpt_config,
    )

    mock_custom_object_save = MagicMock()
    monkeypatch.setattr("verl.utils.checkpoint.fsdp_checkpoint_manager.custom_object_save", mock_custom_object_save)

    manager.save_checkpoint(local_path=str(tmp_path), global_step=0)

    mock_custom_object_save.assert_called_once()
    saved_obj = mock_custom_object_save.call_args.args[0]
    assert saved_obj is peft_model.get_base_model(), "custom_object_save must receive the base model"
    assert saved_obj is not peft_model, "custom_object_save must not receive the PeftModel wrapper"
