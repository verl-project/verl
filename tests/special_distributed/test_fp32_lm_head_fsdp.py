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
import os

import torch
import torch.distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.models.transformers.lm_head import install_fp32_lm_head
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2


def _build_model(tie_word_embeddings: bool, fused: bool) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=127,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
        attention_dropout=0.0,
    )
    model = Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    output_weight = model.lm_head.weight
    install_fp32_lm_head(model)
    if fused:
        apply_monkey_patch(
            model,
            use_remove_padding=False,
            use_fused_kernels=True,
            fused_kernels_backend="torch",
            lm_head_dtype="float32",
        )
    assert model.lm_head.weight is output_weight
    if tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight
    return model


def _forward_value(model, input_ids: torch.Tensor, fused: bool) -> torch.Tensor:
    output = model(
        input_ids=input_ids,
        labels=input_ids if fused else None,
        return_dict=True,
        use_cache=False,
    )
    return output.log_probs if fused else output.logits


def _wrap(model: Qwen3ForCausalLM, strategy: str, mesh):
    if strategy == "fsdp":
        return FSDP(
            model,
            use_orig_params=False,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            device_mesh=mesh,
        )
    if strategy == "fsdp2":
        apply_fsdp2(
            model,
            {
                "mesh": mesh,
                "mp_policy": MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    cast_forward_inputs=True,
                ),
            },
            {},
        )
        return model
    raise ValueError(f"Unknown strategy: {strategy}")


def _full_state_dict(model, strategy: str) -> dict:
    if strategy == "fsdp":
        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            return {key: value.clone() for key, value in model.state_dict().items()}
    options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
    return {key: value.clone() for key, value in get_model_state_dict(model, options=options).items()}


def _load_full_state_dict(model, strategy: str, state_dict: dict):
    if strategy == "fsdp":
        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            model.load_state_dict(state_dict)
        return
    options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
    set_model_state_dict(model, state_dict, options=options)


def test_fp32_lm_head_fsdp():
    if get_device_name() != "cuda":
        print("test_fp32_lm_head_fsdp skipped: FP32 lm_head is CUDA-specific")
        return
    assert get_torch_device().device_count() >= 2, "need at least 2 GPUs for test"

    strategy = os.environ.get("STRATEGY", "fsdp")
    tie_word_embeddings = os.environ.get("TIE_WORD_EMBEDDINGS", "false").lower() == "true"
    fused = os.environ.get("FUSED", "false").lower() == "true"

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    unwrapped_model = _build_model(tie_word_embeddings, fused)
    reference_model = copy.deepcopy(unwrapped_model)
    model = _wrap(unwrapped_model, strategy, mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(100 + dist.get_rank())
    input_ids = torch.randint(0, 127, (2, 31), device="cuda")
    gathered_input_ids = [torch.empty_like(input_ids) for _ in range(world_size)]
    dist.all_gather(gathered_input_ids, input_ids)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        before = _forward_value(model, input_ids, fused)
        loss = before.float().square().mean()
    assert before.dtype == torch.float32
    loss.backward()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_output = _forward_value(reference_model, torch.cat(gathered_input_ids), fused)
        reference_output.float().square().mean().backward()
    if strategy == "fsdp2" and fused and not tie_word_embeddings:
        distributed_grad = model.lm_head.weight.grad.full_tensor()
        torch.testing.assert_close(distributed_grad, reference_model.lm_head.weight.grad, atol=5e-3, rtol=5e-3)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm) and grad_norm > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        after = _forward_value(model, input_ids, fused)
    assert after.dtype == torch.float32
    assert not torch.equal(before, after)

    saved_state = _full_state_dict(model, strategy)
    if dist.get_rank() == 0:
        assert any(key.endswith("lm_head.weight") for key in saved_state), tuple(saved_state)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _forward_value(model, torch.roll(input_ids, 1), fused).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _load_full_state_dict(model, strategy, saved_state)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        restored = _forward_value(model, input_ids, fused)
    torch.testing.assert_close(restored, after, atol=0.0, rtol=0.0)

    if dist.get_rank() == 0:
        print(
            f"FP32 lm_head {strategy=} {tie_word_embeddings=} {fused=} world_size={world_size}: "
            "forward/backward/optimizer/checkpoint passed"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_fp32_lm_head_fsdp()
