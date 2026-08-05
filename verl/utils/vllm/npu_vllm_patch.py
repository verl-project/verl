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


def _patch_legacy_fused_moe_weight_loader(fused_moe) -> bool:
    """Install the NPU transpose wrapper on vLLM's legacy FusedMoE class.

    Legacy vLLM releases expose ``FusedMoE`` as a class whose
    ``weight_loader`` can be wrapped at class level. Modular vLLM releases
    expose ``FusedMoE`` as a factory function, so this class-level patch is
    not applicable. Loader and layout handling for constructed modules is
    owned by their runtime and backend-specific weight-loading paths.
    """
    weight_loader = getattr(fused_moe, "weight_loader", None)
    if not isinstance(fused_moe, type) or not callable(weight_loader):
        return False

    if getattr(weight_loader, "_verl_npu_weight_loader_patched", False):
        return True

    wrapped_weight_loader = vllm_v013_weight_loader_method_wrapper(weight_loader)
    wrapped_weight_loader._verl_npu_weight_loader_patched = True
    fused_moe.weight_loader = wrapped_weight_loader
    return True


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


def apply_npu_vllm_patches() -> None:
    """Apply NPU-specific vLLM patches for weight loading and rotary embedding.

    Must be called before the vLLM engine is created.
    """
    if not is_torch_npu_available(check_device=False):
        return

    # Disable flash_attn in RotaryEmbedding (NPU)
    from vllm.model_executor.layers.fused_moe import FusedMoE

    patch_vllm013_rotary_emb()
    _patch_legacy_fused_moe_weight_loader(FusedMoE)
