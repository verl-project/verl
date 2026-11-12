# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Convert HF weights into a v2 MCore model checkpoint using Megatron-Bridge."""

import argparse
from math import lcm
from pathlib import Path

import torch
import torch.distributed as dist

from verl.models.mcore.bridge import AutoBridge
from verl.models.mcore.bridge_checkpoint import (
    build_conversion_model,
    initialize_conversion,
    load_conversion_checkpoint,
)
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.megatron.dist_checkpointing import save_dist_checkpointing
from verl.utils.megatron_utils import unwrap_model


def _init_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_model_path", required=True, help="HF model path or repository ID")
    parser.add_argument("--output_path", required=True, help="Output MCore weights directory")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size (1 infers from world size)")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--etp_size", type=int, default=1, help="Expert tensor parallel size")
    parser.add_argument("--use_cpu_initialization", action="store_true")
    parser.add_argument("--test", action="store_true", help="Reload saved weights and compare all parameters")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def convert_hf_to_mcore(
    hf_model_path,
    output_path,
    pp_size=1,
    ep_size=1,
    use_cpu_initialization=False,
    test=False,
    trust_remote_code=False,
    *,
    tp_size=1,
    etp_size=1,
):
    output = Path(output_path)
    checkpoint = output / "model" / "dist_ckpt"
    hf_artifacts = output / "model" / "huggingface"
    output_exists = output.is_dir() and any(output.iterdir())
    if output_exists:
        if not (checkpoint / ".metadata").is_file() or not (hf_artifacts / "config.json").is_file():
            raise FileExistsError(
                f"Output path {output_path} is non-empty but is not a complete v2 model checkpoint; "
                "choose an empty output directory"
            )
        if not test:
            print(f"V2 model checkpoint already exists at {output_path}, skipping conversion")
            return
    initialize_conversion()
    world_size = dist.get_world_size()
    if min(tp_size, ep_size, etp_size) < 1:
        raise ValueError("tp_size, ep_size and etp_size must be positive")
    if pp_size == 1:
        parallel_size = lcm(tp_size, etp_size * ep_size)
        if world_size % parallel_size:
            raise ValueError(f"WORLD_SIZE={world_size} must be divisible by lcm(TP, ETP * EP)={parallel_size}")
        pp_size = world_size // parallel_size
    bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=trust_remote_code)
    models = build_conversion_model(
        bridge, pp_size, ep_size, use_cpu_initialization, tp_size=tp_size, etp_size=etp_size
    )
    bridge.load_hf_weights(models)
    model = unwrap_model(models[0])
    if not output_exists:
        checkpoint.mkdir(parents=True, exist_ok=True)
        save_dist_checkpointing({"model": model.sharded_state_dict()}, str(checkpoint))
        if dist.get_rank() == 0:
            bridge.hf_pretrained.save_artifacts(hf_artifacts, original_source_path=hf_model_path)
        dist.barrier()
    if test:
        expected = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        load_conversion_checkpoint(models, checkpoint)
        mismatches = [
            name for name, param in model.named_parameters() if not torch.equal(param.detach().cpu(), expected[name])
        ]
        count = torch.tensor(len(mismatches), device=f"{get_device_name()}:{get_torch_device().current_device()}")
        dist.all_reduce(count)
        if count.item():
            raise AssertionError(f"Checkpoint reload differs from HF-loaded Bridge weights: {mismatches[:20]}")
        print(f"Conversion test passed! rank={dist.get_rank()}, parameters={len(expected)}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    try:
        convert_hf_to_mcore(**vars(_init_args()))
    finally:
        if dist.is_initialized():
            from megatron.core import parallel_state

            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
