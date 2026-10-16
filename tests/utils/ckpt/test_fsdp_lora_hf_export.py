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

import copy
import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM, Qwen2Config

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.fsdp_utils import apply_fsdp2
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def _export_worker(rank, world_size, rendezvous, output_dir, dtype, offload):
    get_torch_device().set_device(rank)
    backend = f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}"
    dist.init_process_group(backend, init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    device = torch.device(get_device_name(), rank)
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").to(device)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.uniform_(-0.1, 0.1)
    reference = copy.deepcopy(model).merge_and_unload().eval()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    kwargs = dict(
        mesh=init_device_mesh(get_device_name(), (world_size,)),
        mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
    )
    if offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()
    apply_fsdp2(model, kwargs, {})

    # Exercise the engine's real export callback with a locally constructed model.
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.module = model
    engine._is_lora = True
    engine._is_offload_param = False
    manager = FSDPCheckpointManager(
        model=model,
        checkpoint_config={"save_contents": ["hf_model"], "load_contents": ["model"]},
        per_tensor_param_fn=engine._get_hf_export_per_tensor_param,
        hf_export_dtype=dtype,
    )
    manager.save_checkpoint(output_dir, global_step=1)

    for name, value in model.state_dict().items():
        actual = value.full_tensor().to(device) if hasattr(value, "full_tensor") else value
        torch.testing.assert_close(actual, original[name], rtol=0, atol=0)

    if rank == 0:
        hf_path = f"{output_dir}/huggingface"
        with open(f"{hf_path}/model.safetensors.index.json") as handle:
            index = json.load(handle)
        saved = {}
        for shard in set(index["weight_map"].values()):
            saved.update(load_file(f"{hf_path}/{shard}"))
        assert set(saved) == set(reference.state_dict()) - {"lm_head.weight"}
        for name, value in saved.items():
            expected = reference.state_dict()[name].to(dtype).cpu()
            torch.testing.assert_close(value, expected, msg=lambda message, name=name: f"{name}: {message}")
        loaded = AutoModelForCausalLM.from_pretrained(hf_path, dtype=dtype, attn_implementation="eager").to(device)
        inputs = torch.arange(16, device=device).view(1, -1)
        with torch.no_grad():
            torch.testing.assert_close(loaded(inputs).logits, copy.deepcopy(reference).to(dtype)(inputs).logits)

    # Rollout keeps its BF16 contract, including after an HF export.
    rollout_weights = dict(engine._merged_lora_per_tensor_param())
    for name, value in rollout_weights.items():
        torch.testing.assert_close(
            value, reference.state_dict()[name].to(torch.bfloat16), msg=lambda message, name=name: f"{name}: {message}"
        )
    dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("offload", [False, True])
def test_fsdp2_lora_hf_export(tmp_path, dtype, offload):
    mp.spawn(
        _export_worker,
        args=(2, str(tmp_path / "rendezvous"), str(tmp_path / "checkpoint"), dtype, offload),
        nprocs=2,
        join=True,
    )
