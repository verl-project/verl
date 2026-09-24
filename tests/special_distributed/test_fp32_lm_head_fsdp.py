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

import torch
import torch.distributed as dist
from torch.distributed import init_device_mesh
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.models.transformers.lm_head import install_fp32_lm_head
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2


def _build_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=127,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    install_fp32_lm_head(model)
    apply_monkey_patch(
        model,
        use_remove_padding=False,
        use_fused_kernels=True,
        fused_kernels_backend="torch",
        lm_head_dtype="float32",
    )
    return model


def main():
    if get_device_name() != "cuda":
        print("test skipped: FP32 lm_head is CUDA-specific")
        return
    assert get_torch_device().device_count() >= 2, "need at least 2 GPUs for test"

    _, _, world_size = initialize_global_process_group()
    torch.manual_seed(42)

    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    model = _build_model()
    reference_model = copy.deepcopy(model)
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

    torch.manual_seed(100 + dist.get_rank())
    input_ids = torch.randint(0, 127, (2, 31), device="cuda")
    gathered_input_ids = [torch.empty_like(input_ids) for _ in range(world_size)]
    dist.all_gather(gathered_input_ids, input_ids)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids=input_ids, labels=input_ids, return_dict=True, use_cache=False).log_probs
        output.square().mean().backward()
        reference_ids = torch.cat(gathered_input_ids)
        reference_output = reference_model(
            input_ids=reference_ids,
            labels=reference_ids,
            return_dict=True,
            use_cache=False,
        ).log_probs
        reference_output.square().mean().backward()

    distributed_grad = model.lm_head.weight.grad.full_tensor()
    torch.testing.assert_close(distributed_grad, reference_model.lm_head.weight.grad, atol=5e-3, rtol=5e-3)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
