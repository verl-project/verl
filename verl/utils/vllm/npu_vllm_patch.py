# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
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


from functools import wraps

from verl.utils.device import is_torch_npu_available


def vllm_v013_weight_loader_method_wrapper(fn):
    @wraps(fn)
    def wrapper(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        if (shard_id in ("w1", "w3") and param.shape[1] == self.hidden_size) or (
            shard_id == "w2" and param.shape[2] == self.hidden_size
        ):
            param.data = param.data.transpose(1, 2)
        return fn(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success)

    return wrapper


def patch_vllm013_rotary_emb():
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    def vllm013_npu_rotary_embedding_init_impl(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> None:
        super(ApplyRotaryEmb, self).__init__()
        self.is_neox_style = is_neox_style
        self.enable_fp32_compute = enable_fp32_compute
        self.apply_rotary_emb_flash_attn = None

    ApplyRotaryEmb.__init__ = vllm013_npu_rotary_embedding_init_impl


if is_torch_npu_available(check_device=False):
    import vllm
    from packaging import version

    _VLLM_VERSION = version.parse(vllm.__version__)
    if _VLLM_VERSION >= version.parse("0.13.0") and _VLLM_VERSION <= version.parse("0.14.0"):
        # Disable flash_attn in RotaryEmbedding (NPU) when VLLM >= 0.13
        from vllm.model_executor.layers.fused_moe import FusedMoE

        patch_vllm013_rotary_emb()
        FusedMoE.weight_loader = vllm_v013_weight_loader_method_wrapper(FusedMoE.weight_loader)
    elif _VLLM_VERSION >= version.parse("0.18.0"):
        # Disable flash_attn in RotaryEmbedding (NPU) when VLLM >= 0.18
        from vllm.model_executor.layers.fused_moe import FusedMoE

        patch_vllm013_rotary_emb()
        FusedMoE.weight_loader = vllm_v013_weight_loader_method_wrapper(FusedMoE.weight_loader)
