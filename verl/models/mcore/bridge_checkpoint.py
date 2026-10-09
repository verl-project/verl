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

"""Bridge model construction shared by the HF/MCore checkpoint tools."""

import os
from pathlib import Path

import torch
import torch.distributed as dist

from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device


def get_dynamic_pipeline_shards(layer_num: int, pp_size: int) -> list[int]:
    """Calculate the pipeline sharding configuration for Megatron-LM.

    Args:
        layer_num: Total number of layers in the model.
        pp_size: Number of pipeline parallel ranks.

    Returns:
        layer number of each pp rank. Make the sharding of the pipeline as uniform as possible.
    """
    if layer_num < pp_size:
        raise ValueError(f"layer_num {layer_num} must be greater than pp_size {pp_size}.")

    if pp_size < 1:
        raise ValueError(f"pp_size must be at least 1, got {pp_size}.")
    if pp_size == 1:
        return [layer_num]

    if pp_size == 2:
        return [
            layer_num // 2,
            layer_num - layer_num // 2,
        ]

    middle_size = pp_size - 2
    shards_strategy = []
    for middle_layer_num in range(layer_num):
        first_last_layer_num = layer_num - middle_layer_num * middle_size
        first_layer_num = first_last_layer_num // 2
        last_layer_num = first_last_layer_num - first_last_layer_num // 2
        if 0 < first_layer_num <= middle_layer_num and 0 < last_layer_num <= middle_layer_num:
            shards_strategy.append(
                (
                    [first_layer_num] + [middle_layer_num] * middle_size + [last_layer_num],
                    abs(first_layer_num - middle_layer_num),
                )
            )

    # sort by diff of layer_num, to make it as uniform as possible
    res = sorted(shards_strategy, key=lambda x: x[1])[0][0]
    assert sum(res) == layer_num, f"sum(res)={sum(res)} != layer_num={layer_num}, pp_size={pp_size}"
    return res


def initialize_conversion():
    """Initialize torchrun (or a single local rank) before constructing a model."""
    for name, default in dict(RANK="0", WORLD_SIZE="1", MASTER_ADDR="localhost", MASTER_PORT="12355").items():
        os.environ.setdefault(name, default)
    get_torch_device().set_device(f"{get_device_name()}:{os.environ.get('LOCAL_RANK', '0')}")
    if not dist.is_initialized():
        dist.init_process_group(get_nccl_backend())


def build_conversion_model(
    bridge, pp_size, ep_size=1, use_cpu_initialization=False, is_value_model=False, *, tp_size=1, etp_size=1
):
    """Use the same Bridge provider and value-head hook as the training engine.

    Conversion defaults to TP=1 and ETP=1 with no virtual pipeline. Distributed
    checkpoints can be resharded to the training engine's parallelism when loaded.
    """
    from verl.models.mcore.bridge import make_value_model

    provider = bridge.to_megatron_provider(load_weights=False)
    world_size = dist.get_world_size()
    if min(tp_size, pp_size, ep_size, etp_size) < 1:
        raise ValueError("tp_size, pp_size, ep_size and etp_size must be positive")
    if world_size % (tp_size * pp_size) or world_size % (etp_size * ep_size * pp_size):
        raise ValueError(
            f"WORLD_SIZE={world_size} must be divisible by TP * PP ({tp_size * pp_size}) "
            f"and ETP * EP * PP ({etp_size * ep_size * pp_size})"
        )
    shards = get_dynamic_pipeline_shards(provider.num_layers, pp_size)
    overrides = dict(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
        sequence_parallel=tp_size > 1,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        fp16=False,
        use_cpu_initialization=use_cpu_initialization,
        num_layers_in_first_pipeline_stage=shards[0] if pp_size > 1 else None,
        num_layers_in_last_pipeline_stage=shards[-1] if pp_size > 2 else None,
    )
    if is_value_model:
        overrides["share_embeddings_and_output_weights"] = False
    for key, value in overrides.items():
        setattr(provider, key, value)
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    if is_value_model:
        provider.register_pre_wrap_hook(make_value_model(provider.hidden_size, provider.sequence_parallel))
    models = provider.provide_distributed_model(
        wrap_with_ddp=False, bf16=True, use_cpu_initialization=use_cpu_initialization
    )
    return models


def load_conversion_checkpoint(models, checkpoint_dir):
    """Load model state from a v2 ``model/dist_ckpt`` subtree."""
    from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing
    from verl.utils.megatron_utils import unwrap_model

    if not (Path(checkpoint_dir) / ".metadata").is_file():
        raise FileNotFoundError(f"No distributed checkpoint metadata found in {checkpoint_dir}")
    modules = [unwrap_model(model) for model in models]
    state = {
        f"model{i}" if len(modules) > 1 else "model": model.sharded_state_dict() for i, model in enumerate(modules)
    }
    loaded = load_dist_checkpointing(state, str(checkpoint_dir))
    for i, model in enumerate(modules):
        model.load_state_dict(loaded[f"model{i}" if len(modules) > 1 else "model"], strict=True)


def compare_hf_weights(weights, reference_path, *, atol=1e-2, rtol=5e-2):
    """Compare a collective Bridge export against every tensor in a sharded HF model.

    Drain the generator on every rank even after a mismatch: export performs
    collectives. Broadcast the final result so failures cannot strand other ranks.
    """
    from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

    source = SafeTensorsStateSource(reference_path)
    expected = set(source.keys())
    seen = set()
    errors = []
    for name, tensor in weights:
        if dist.get_rank() != 0:
            continue
        if name not in expected:
            errors.append(f"Unexpected exported parameter: {name}")
            continue
        seen.add(name)
        try:
            reference = source[name]
            torch.testing.assert_close(tensor.cpu(), reference.to(tensor.dtype), atol=atol, rtol=rtol)
        except (AssertionError, RuntimeError, ValueError) as error:
            errors.append(f"{name}: {error}")
    if dist.get_rank() == 0:
        errors.extend(f"Missing exported parameter: {name}" for name in sorted(expected - seen))
    result = [errors[:20] if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(result, src=0)
    if result[0]:
        raise AssertionError("HF weight comparison failed:\n" + "\n".join(result[0]))
    if dist.get_rank() == 0:
        print(f"HF comparison passed: {len(seen)} tensors", flush=True)
