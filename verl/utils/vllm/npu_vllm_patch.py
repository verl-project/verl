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
from collections.abc import Iterable
from functools import wraps

import torch

from verl.utils.device import is_torch_npu_available

logger = logging.getLogger(__name__)

_GLM52_PATCH_ENV = "VERL_VLLM_ASCEND_GLM52_PATCH"
_GLM52_PATCH_TRUTHY_VALUES = {"1", "true", "yes"}
_GLM52_PATCH_MARKER = "_verl_vllm_ascend_glm52_patched"


def normalize_deepseek_v4_weight_names(
    model: torch.nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[tuple[str, torch.Tensor]]:
    """Translate Transformers DeepSeek V4 names for the Ascend weight loader."""
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "deepseek_v4":
        return weights

    replacements = (
        (".self_attn.q_a_proj.", ".self_attn.wq_a."),
        (".self_attn.kv_proj.", ".self_attn.wkv."),
        (".self_attn.kv_a_proj_with_mqa.", ".self_attn.wkv."),
        (".self_attn.q_b_proj.", ".self_attn.wq_b."),
        (".self_attn.o_a_proj.", ".self_attn.wo_a."),
        (".self_attn.o_b_proj.", ".self_attn.wo_b."),
        (".self_attn.q_a_norm.", ".self_attn.q_norm."),
        (".self_attn.sinks", ".self_attn.attn_sink"),
        (".self_attn.compressor.kv_proj.", ".self_attn.compressor.wkv."),
        (".self_attn.compressor.gate_proj.", ".self_attn.compressor.wgate."),
        (".self_attn.compressor.kv_norm.", ".self_attn.compressor.norm."),
        (".self_attn.compressor.position_bias", ".self_attn.compressor.ape"),
        (".self_attn.compressor.indexer.position_bias", ".self_attn.indexer.compressor.ape"),
        (".self_attn.compressor.indexer.kv_proj.", ".self_attn.indexer.compressor.wkv."),
        (".self_attn.compressor.indexer.gate_proj.", ".self_attn.indexer.compressor.wgate."),
        (".self_attn.compressor.indexer.q_b_proj.", ".self_attn.indexer.wq_b."),
        (".self_attn.compressor.indexer.kv_norm.", ".self_attn.indexer.compressor.norm."),
        (".self_attn.compressor.indexer.", ".self_attn.indexer."),
        (".self_attn.indexer.scorer.weights_proj.", ".self_attn.indexer.weights_proj."),
        ("hc_head.hc_fn", "hc_head_fn"),
        ("hc_head.hc_base", "hc_head_base"),
        ("hc_head.hc_scale", "hc_head_scale"),
        (".attn_hc.fn", ".hc_attn_fn"),
        (".attn_hc.base", ".hc_attn_base"),
        (".attn_hc.scale", ".hc_attn_scale"),
        (".ffn_hc.fn", ".hc_ffn_fn"),
        (".ffn_hc.base", ".hc_ffn_base"),
        (".ffn_hc.scale", ".hc_ffn_scale"),
    )

    def normalized_weights():
        for name, tensor in weights:
            for source, target in replacements:
                name = name.replace(source, target)
            yield name, tensor

    return normalized_weights()


def _filter_deepseek_v4_layer_weights(
    model: torch.nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[tuple[str, torch.Tensor]]:
    """Skip checkpoint layers beyond the constructed DeepSeek V4 model."""
    config = getattr(model, "config", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if getattr(config, "model_type", None) != "deepseek_v4" or layers is None:
        yield from weights
        return

    num_layers = len(layers)
    for name, tensor in weights:
        prefixed_name = name if name.startswith("model") else f"model.{name}"
        if prefixed_name.startswith("model.layers."):
            layer_field = prefixed_name[len("model.layers.") :].split(".", 1)[0]
            if layer_field.isdigit() and int(layer_field) >= num_layers:
                continue
        yield name, tensor


def _patch_deepseek_v4_weight_loader(model_class) -> bool:
    original_load_weights = model_class.load_weights
    if getattr(original_load_weights, "_verl_deepseek_v4_weight_names_patched", False):
        return False

    @wraps(original_load_weights)
    def load_weights(self, weights, *args, **kwargs):
        weights = normalize_deepseek_v4_weight_names(self, weights)
        weights = _filter_deepseek_v4_layer_weights(self, weights)
        return original_load_weights(self, weights, *args, **kwargs)

    load_weights._verl_deepseek_v4_weight_names_patched = True
    model_class.load_weights = load_weights
    return True


def patch_vllm_ascend_deepseek_v4_weight_loader() -> None:
    """Patch DeepSeek V4 initialization and loading without editing vllm-ascend."""
    from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM

    _patch_deepseek_v4_weight_loader(AscendDeepseekV4ForCausalLM)


def patch_vllm_ascend_v023_sched_yield():
    """Disable sched_yield in vllm.distributed.utils on ARM hosts.

    On ARM hosts (e.g. Atlas A2 machines) the scheduler busy-wait with
    sched_yield degrades badly, so USE_SCHED_YIELD must stay disabled.
    Drop this patch once the fix is included in the installed vllm-ascend release.
    """
    import sys

    import vllm.distributed.utils
    from vllm.platforms import CpuArchEnum, Platform

    is_arm = Platform.get_cpu_architecture() == CpuArchEnum.ARM

    use_sched_yield = (
        (sys.version_info[:3] >= (3, 11, 1)) or (sys.version_info[:2] == (3, 10) and sys.version_info[2] >= 8)
    ) and not is_arm

    vllm.distributed.utils.USE_SCHED_YIELD = use_sched_yield


def _glm52_patch_enabled() -> bool:
    return os.getenv(_GLM52_PATCH_ENV, "").strip().lower() in _GLM52_PATCH_TRUTHY_VALUES


def _normalized_source(fn) -> str:
    try:
        return "".join(inspect.getsource(fn).split())
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"{_GLM52_PATCH_ENV} requires inspectable vllm/vllm-ascend 0.23.x Python sources.") from exc


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


def patch_vllm_ascend_glm52() -> None:
    """Apply the vllm-ascend GLM-5.2 fixes from commit ca0420391."""
    try:
        from vllm_ascend.ops import layernorm
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(f"{_GLM52_PATCH_ENV} could not load the required vllm-ascend 0.23.x patch targets.") from exc

    patched = [
        _patch_glm52_ascend_rms_norm(layernorm),
    ]
    status = "applied" if any(patched) else "already present"
    logger.info("GLM-5.2 vllm-ascend compatibility patch %s", status)


def patch_vllm_ascend_v023_fia_decode_mask():
    import inspect
    import textwrap

    # Import vllm_ascend.ops first to break a circular import in vllm-ascend:
    # attention_v1 -> device_op -> vllm_ascend.ops.* -> fused_moe -> device_op.
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.attention import attention_v1

    # Each entry is an (old, new) pair applied to the inspect.getsource +
    # textwrap.dedent'ed source of the method (def line at 0, body at 4),
    # mirroring the upstream diff exactly.
    replacements = {
        "full_graph_fia": [
            (
                "    attn_mask = attn_metadata.attn_mask\n"
                "    sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0\n",
                "    # attn_state is forced to DecodeOnly during FULL graph capture, including\n"
                "    # mixed prefill-decode graphs and spec-decode verify batches, which still\n"
                "    # need the causal mask. Only genuine single-token decode may skip the mask;\n"
                "    # max_query_len is 1 for pure decode and for uniform-decode capture, but\n"
                "    # equals num_tokens for mixed-batch capture and 1 + num_spec_tokens for\n"
                "    # spec-decode capture.\n"
                "    is_decode = attn_metadata.attn_state == AscendAttentionState.DecodeOnly"
                " and attn_metadata.max_query_len == 1\n"
                "    attn_mask, sparse_mode = (\n"
                "        (None, 0)\n"
                "        if is_decode and not self.sliding_window\n"
                "        else (attn_metadata.attn_mask, 4 if self.sliding_window else 3"
                " if attn_metadata.causal else 0)\n"
                "    )\n",
            )
        ],
        "forward_fused_infer_attention": [
            (
                "            attn_output, _ = DeviceOperator.npu_fused_infer_attention_score(\n"
                "                query=query,\n"
                "                key=key,\n"
                "                value=value,\n"
                "                atten_mask=attn_metadata.attn_mask,\n",
                "            is_decode = (\n"
                "                attn_metadata.attn_state == AscendAttentionState.DecodeOnly"
                " and attn_metadata.max_query_len == 1\n"
                "            )\n"
                "            attn_output, _ = DeviceOperator.npu_fused_infer_attention_score(\n"
                "                query=query,\n"
                "                key=key,\n"
                "                value=value,\n"
                "                atten_mask=None if is_decode else attn_metadata.attn_mask,\n",
            ),
            (
                "                is_prefill_no_cache=attn_metadata.attn_state == AscendAttentionState.PrefillNoCache,\n"
                "                sparse_mode=3,\n",
                "                is_prefill_no_cache=attn_metadata.attn_state == AscendAttentionState.PrefillNoCache,\n"
                "                sparse_mode=0 if is_decode else 3,\n",
            ),
        ],
    }

    module_globals = vars(attention_v1)
    for method_name, method_replacements in replacements.items():
        src = textwrap.dedent(inspect.getsource(getattr(attention_v1.AscendAttentionBackendImpl, method_name)))
        if "is_decode" in src:
            # The installed vllm-ascend already contains the fix.
            continue
        try:
            for old, new in method_replacements:
                assert src.count(old) == 1
                src = src.replace(old, new, 1)
        except AssertionError:
            logger.warning(
                "Skip patching AscendAttentionBackendImpl.%s: the installed vllm-ascend source "
                "does not match the expected releases/v0.23.0 code.",
                method_name,
            )
            continue
        assert method_name not in module_globals
        exec(src, module_globals)
        patched_method = module_globals.pop(method_name)
        setattr(attention_v1.AscendAttentionBackendImpl, method_name, patched_method)


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
    """Apply NPU-specific vLLM patches for weight loading, rotary embedding, and sleep.

    Must be called before the vLLM engine is created.
    """
    if not is_torch_npu_available(check_device=False):
        return

    import vllm
    from packaging import version

    _VLLM_VERSION = version.parse(getattr(vllm, "__version__", "0.0.0"))
    if version.parse("0.23.0") <= _VLLM_VERSION < version.parse("0.24.0"):
        # On ARM hosts (e.g. Atlas A2) the scheduler busy-wait with sched_yield degrades badly.
        patch_vllm_ascend_v023_sched_yield()
        # Only genuine single-token decode may skip the causal attention mask; mixed
        # prefill-decode and spec-decode FULL graph capture batches still need it.
        patch_vllm_ascend_v023_fia_decode_mask()
        patch_vllm_ascend_deepseek_v4_weight_loader()

    # Disable flash_attn in RotaryEmbedding (NPU)

    patch_vllm013_rotary_emb()
    if _glm52_patch_enabled():
        import vllm_ascend.patch.worker.patch_routed_experts_capture  # noqa: F401

        patch_vllm_ascend_glm52()
