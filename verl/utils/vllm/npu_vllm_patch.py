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


import inspect
import logging
import os
from contextvars import ContextVar
from functools import wraps

from verl.utils.device import is_torch_npu_available

logger = logging.getLogger(__name__)

_GLM52_PATCH_ENV = "VERL_VLLM_ASCEND_GLM52_PATCH"
_GLM52_PATCH_TRUTHY_VALUES = {"1", "true", "yes"}
_GLM52_PATCH_MARKER = "_verl_vllm_ascend_glm52_patched"


def _glm52_patch_enabled() -> bool:
    return os.getenv(_GLM52_PATCH_ENV, "").strip().lower() in _GLM52_PATCH_TRUTHY_VALUES


def _normalized_source(fn) -> str:
    try:
        return "".join(inspect.getsource(fn).split())
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"{_GLM52_PATCH_ENV} requires inspectable vllm-ascend 0.23.x Python sources.") from exc


def _patch_glm52_ascend_rms_norm(layernorm_module) -> bool:
    ascend_rms_norm = layernorm_module.AscendRMSNorm
    original_forward = ascend_rms_norm.forward_oot
    if getattr(original_forward, _GLM52_PATCH_MARKER, False):
        return False

    source = _normalized_source(original_forward)
    if "residual=x+residual" in source and "npu_rms_norm(residual,self.weight" in source:
        return False
    if "npu_add_rms_norm" not in source:
        raise RuntimeError(f"{_GLM52_PATCH_ENV} found an unsupported AscendRMSNorm.forward_oot implementation.")

    @wraps(original_forward)
    def patched_forward(self, x, residual=None):
        if residual is None:
            return original_forward(self, x, residual)

        import torch_npu

        residual = layernorm_module.torch.ops.vllm.maybe_chunk_residual(x, residual)
        residual = x + residual
        x = torch_npu.npu_rms_norm(residual, self.weight, epsilon=self.variance_epsilon)[0]
        return x, residual

    setattr(patched_forward, _GLM52_PATCH_MARKER, True)
    ascend_rms_norm.forward_oot = patched_forward
    return True


def _patch_glm52_sfa_kv_b_proj_disposal(sfa_module) -> bool:
    ascend_sfa_impl = sfa_module.AscendSFAImpl
    original_process_weights = ascend_sfa_impl.process_weights_after_loading
    if getattr(original_process_weights, _GLM52_PATCH_MARKER, False):
        return False

    source = _normalized_source(original_process_weights)
    if "dispose_layer(self.kv_b_proj)" not in source:
        return False

    original_dispose_layer = sfa_module.dispose_layer
    no_preserved_layer = object()
    preserved_layer: ContextVar[object] = ContextVar(
        "verl_vllm_ascend_glm52_preserved_layer", default=no_preserved_layer
    )

    @wraps(original_dispose_layer)
    def selective_dispose_layer(layer):
        if layer is preserved_layer.get():
            return None
        return original_dispose_layer(layer)

    @wraps(original_process_weights)
    def patched_process_weights(self, act_dtype):
        token = preserved_layer.set(self.kv_b_proj)
        try:
            return original_process_weights(self, act_dtype)
        finally:
            preserved_layer.reset(token)

    setattr(patched_process_weights, _GLM52_PATCH_MARKER, True)
    sfa_module.dispose_layer = selective_dispose_layer
    ascend_sfa_impl.process_weights_after_loading = patched_process_weights
    return True


def _patch_glm52_sfa_indexer_scale(sfa_module) -> bool:
    indexer_post_process = sfa_module.AscendSFAImpl.indexer_select_post_process
    source = _normalized_source(indexer_post_process)
    scale_expression = "weights=weights*(self.n_head**-0.5)*(self.head_dim**-0.5)"
    if scale_expression in source:
        return False
    if "weights=kw[:,self.head_dim:]" not in source or "DeviceOperator.indexer_select_post_process" not in source:
        raise RuntimeError(
            f"{_GLM52_PATCH_ENV} found an unsupported AscendSFAImpl.indexer_select_post_process implementation."
        )

    device_operator = sfa_module.DeviceOperator
    original_operator = device_operator.indexer_select_post_process
    if getattr(original_operator, _GLM52_PATCH_MARKER, False):
        return False

    parameter_names = tuple(inspect.signature(original_operator).parameters)
    expected_prefix = ("sfa_impl", "q_li", "q_li_scale", "q_li_shape_ori", "weights")
    if parameter_names[: len(expected_prefix)] != expected_prefix:
        raise RuntimeError(
            f"{_GLM52_PATCH_ENV} found an unsupported DeviceOperator.indexer_select_post_process signature."
        )

    @wraps(original_operator)
    def patched_operator(sfa_impl, q_li, q_li_scale, q_li_shape_ori, weights, *args, **kwargs):
        weights = weights * (sfa_impl.n_head**-0.5) * (sfa_impl.head_dim**-0.5)
        return original_operator(sfa_impl, q_li, q_li_scale, q_li_shape_ori, weights, *args, **kwargs)

    setattr(patched_operator, _GLM52_PATCH_MARKER, True)
    device_operator.indexer_select_post_process = staticmethod(patched_operator)
    return True


def patch_vllm_ascend_glm52() -> None:
    """Apply the vllm-ascend GLM-5.2 fixes from commit ca0420391."""
    try:
        from vllm_ascend.attention import sfa_v1
        from vllm_ascend.ops import layernorm
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(f"{_GLM52_PATCH_ENV} could not load the required vllm-ascend 0.23.x patch targets.") from exc

    patched = [
        _patch_glm52_ascend_rms_norm(layernorm),
        _patch_glm52_sfa_kv_b_proj_disposal(sfa_v1),
        _patch_glm52_sfa_indexer_scale(sfa_v1),
    ]
    status = "applied" if any(patched) else "already present"
    logger.info("GLM-5.2 vllm-ascend compatibility patch %s", status)


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


def patch_camem_sleep() -> None:
    """Synchronize pending NPU work before CaMem sleep unmaps tensor memory."""
    try:
        import vllm_ascend.device_allocator.camem as camem
    except ModuleNotFoundError as exc:
        if exc.name in ("vllm_ascend", "vllm_ascend.device_allocator", "vllm_ascend.device_allocator.camem"):
            return
        raise

    original_sleep = camem.CaMemAllocator.sleep
    if getattr(original_sleep, "_verl_camem_sleep_patched", False):
        return

    @wraps(original_sleep)
    def patched_sleep(self, *args, **kwargs):
        # Graph replay and normal NPU launches are asynchronous. All users of
        # the virtual mappings must finish before sleep calls unmap_and_release;
        # synchronizing via empty_cache() after the unmap loop is too late.
        camem.torch.npu.synchronize()
        return original_sleep(self, *args, **kwargs)

    patched_sleep._verl_camem_sleep_patched = True
    camem.CaMemAllocator.sleep = patched_sleep


def apply_npu_vllm_patches() -> None:
    """Apply NPU-specific vLLM patches for weight loading, rotary embedding, and sleep.

    Must be called before the vLLM engine is created.
    """
    if not is_torch_npu_available(check_device=False):
        return

    # Disable flash_attn in RotaryEmbedding (NPU)
    from vllm.model_executor.layers import fused_moe

    patch_vllm013_rotary_emb()
    _patch_legacy_fused_moe_weight_loader(getattr(fused_moe, "FusedMoE", None))
    if _glm52_patch_enabled():
        import vllm_ascend.patch.worker.patch_routed_experts_capture  # noqa: F401

        patch_camem_sleep()
        patch_vllm_ascend_glm52()
