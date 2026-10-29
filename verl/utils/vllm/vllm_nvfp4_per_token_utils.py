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
"""vLLM rollout for the ``NVFP4_PER_TOKEN`` routed-expert preset.

The trainer exports compressed-tensors NVFP4 expert weights; vLLM loads them and
quantizes activations per token in FlashInfer's TRT-LLM kernel, matching
Transformer Engine's row-scaled NVFP4 training. vLLM supports per-token
activations for its online method and ModelOpt checkpoints; this module adds
the compressed-tensors equivalent.

Refits go through the NVFP4 refit in ``vllm_fp4_utils``; this module adds the
per-token checks around it.
"""

import re
from dataclasses import replace
from typing import Optional

import torch

from verl.utils.qat.compressed_tensors import (
    is_nvfp4_per_token_config,
    is_nvfp4_per_token_scheme,
    nvfp4_per_token_module_matcher,
)

_EXPERT_TENSOR_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(\w+)$")
_QUANTIZED_SUFFIXES = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
_PACKED_EXPERT_TENSOR_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)(?:\.weight)?$")
_MOE_MODULE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts(?:\.routed_experts)?$")


def install_nvfp4_per_token_compressed_tensors_patch():
    """Route ``NVFP4_PER_TOKEN`` compressed-tensors experts to the per-token TRT-LLM kernel."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend, make_nvfp4_moe_kernel
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
        compressed_tensors_moe_w4a4_nvfp4,
    )

    if getattr(CompressedTensorsMoEMethod, "_verl_nvfp4_per_token_method", None) is not None:
        return
    base = compressed_tensors_moe_w4a4_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod

    class CompressedTensorsW4A4Nvfp4PerTokenMoEMethod(base):
        """Compressed-tensors NVFP4 weights with per-token dynamic NVFP4 activations."""

        def __init__(self, moe, layer_name=None):
            # verl runs per-token activations on FlashInfer TRT-LLM.
            selection_moe = replace(moe, moe_backend="flashinfer_trtllm") if moe.moe_backend == "auto" else moe
            super().__init__(selection_moe, layer_name=layer_name, use_a16=False)
            if self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM:
                raise NotImplementedError(
                    f"NVFP4_PER_TOKEN requires the FlashInfer TRT-LLM MoE backend, got {self.nvfp4_backend}"
                )

        def process_weights_after_loading(self, layer):
            # The base method with per-token activations. The conversion helpers are looked up on
            # the base method's module, where the NVFP4 refit in vllm_fp4_utils redirects them.
            layer.w13_weight = torch.nn.Parameter(layer.w13_weight_packed.data, requires_grad=False)
            delattr(layer, "w13_weight_packed")
            layer.w2_weight = torch.nn.Parameter(layer.w2_weight_packed.data, requires_grad=False)
            delattr(layer, "w2_weight_packed")

            # gate_proj and up_proj share one global scale and the static input scales
            # are 1; NVFP4PerTokenRefitCheck enforces both on every refit (the dummy
            # initial load carries neither). Per-token activation scales come from the kernel.
            ones = torch.ones(layer.num_experts, device=layer.w13_weight.device, dtype=torch.float32)
            (w13, w13_scale, w13_scale_2, a13_scale, w2, w2_scale, w2_scale_2, a2_scale) = (
                compressed_tensors_moe_w4a4_nvfp4.convert_to_nvfp4_moe_kernel_format(
                    nvfp4_backend=self.nvfp4_backend,
                    layer=layer,
                    w13=layer.w13_weight,
                    w13_scale=layer.w13_weight_scale,
                    w13_scale_2=(1.0 / layer.w13_weight_global_scale[:, 0]).contiguous(),
                    a13_scale=ones,
                    w2=layer.w2_weight,
                    w2_scale=layer.w2_weight_scale,
                    w2_scale_2=1.0 / layer.w2_weight_global_scale,
                    a2_scale=ones.clone(),
                    is_act_and_mul=self.moe.is_act_and_mul,
                )
            )
            replace_parameter = compressed_tensors_moe_w4a4_nvfp4.replace_parameter
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
            replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
            replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
            layer.w13_input_scale = a13_scale
            layer.w2_input_scale = a2_scale

            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            self.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                backend=self.nvfp4_backend,
                routing_tables=layer._expert_routing_tables(),
                per_token_activation=True,
            )
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)
            layer._already_called_process_weights_after_loading = True

    original = CompressedTensorsMoEMethod.get_moe_method

    def get_moe_method(quant_config, layer, layer_name):
        quant_config._add_fused_moe_to_target_scheme_map()
        schemes = [
            quant_config.get_scheme_dict(layer, f"{layer_name}.0.{projection}")
            for projection in (layer.ckpt_gate_proj_name, layer.ckpt_up_proj_name, layer.ckpt_down_proj_name)
        ]
        if schemes[0] is not None and all(scheme == schemes[0] for scheme in schemes):
            weights, inputs = schemes[0].get("weights"), schemes[0].get("input_activations")
            if is_nvfp4_per_token_scheme(weights, inputs):
                return CompressedTensorsW4A4Nvfp4PerTokenMoEMethod(layer.moe_config, layer_name)
        return original(quant_config, layer, layer_name)

    CompressedTensorsMoEMethod.get_moe_method = staticmethod(get_moe_method)
    # The installed class doubles as the patch marker and the type that is_nvfp4_per_token_method checks.
    CompressedTensorsMoEMethod._verl_nvfp4_per_token_method = CompressedTensorsW4A4Nvfp4PerTokenMoEMethod


def nvfp4_per_token_quantization_config(vllm_config) -> dict | None:
    """The rollout ``quantization_config`` when any of its config groups is ``NVFP4_PER_TOKEN``.

    Reads the config vLLM built its quantization methods from (checkpoint or ``hf_overrides``),
    falling back to the HF config before vLLM has built one.
    """
    config = getattr(getattr(vllm_config, "quant_config", None), "config", None)
    if config is None:
        hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
        config = getattr(hf_config, "quantization_config", None)
    if not is_nvfp4_per_token_config(config):
        return None
    return config.to_dict() if hasattr(config, "to_dict") else config


def is_nvfp4_per_token_method(quant_method) -> bool:
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
    )

    method_cls = getattr(CompressedTensorsMoEMethod, "_verl_nvfp4_per_token_method", None)
    return method_cls is not None and isinstance(quant_method, method_cls)


def routed_expert_layers(model: torch.nn.Module, quantization_config: dict) -> dict[int, tuple[torch.nn.Module, bool]]:
    """vLLM's routed-expert modules by layer index, each with whether the config quantizes it per token.

    The layers come from the built model and the precision from the export's name matching,
    so no per-architecture layer rule is needed.
    """
    per_token = nvfp4_per_token_module_matcher(quantization_config)
    layers = {}
    for module_name, module in model.named_modules():
        match = _MOE_MODULE_RE.search(module_name)
        if match is None or getattr(module, "quant_method", None) is None:
            continue
        layer = int(match.group(1))
        if layer in layers:
            raise RuntimeError(f"Routed experts of layer {layer} were found twice")
        # vLLM resolves the scheme of a fused MoE from its first expert's projections.
        layers[layer] = (module, per_token(f"{module_name.removesuffix('.routed_experts')}.0.gate_proj"))
    if not any(quantized for _, quantized in layers.values()):
        raise RuntimeError("NVFP4_PER_TOKEN rollout found no routed-expert layer to quantize")
    return layers


class NVFP4PerTokenRefitCheck:
    """Check one refit round: each expert projection arrives once, in the expected precision and layout."""

    def __init__(self, model: torch.nn.Module, quantization_config: dict):
        projections = ("gate_proj", "up_proj", "down_proj")
        self.layers = routed_expert_layers(model, quantization_config)
        self.expected = {
            (layer, expert, projection, suffix)
            for layer, (module, quantized) in self.layers.items()
            for expert in range(module.global_num_experts)
            for projection in projections
            for suffix in (_QUANTIZED_SUFFIXES if quantized else ("weight",))
        }
        self.seen = set()
        self.gate_up_global_scales = {}

    def _keys(self, name: str, tensor: torch.Tensor) -> list[tuple[int, int, str, str]]:
        match = _EXPERT_TENSOR_RE.search(name)
        if match is not None:
            return [(int(match.group(1)), int(match.group(2)), match.group(3), match.group(4))]
        # BF16 experts of packed-expert checkpoints (e.g. Qwen3.5) stay packed: one tensor per layer.
        match = _PACKED_EXPERT_TENSOR_RE.search(name)
        if match is not None:
            projections = ("gate_proj", "up_proj") if match.group(2) == "gate_up_proj" else ("down_proj",)
            layer = int(match.group(1))
            return [(layer, expert, p, "weight") for expert in range(tensor.shape[0]) for p in projections]
        return []

    def _layout(self, layer: int, projection: str, suffix: str) -> tuple[tuple[int, ...], Optional[torch.dtype]]:
        """Checkpoint-layout shape and dtype of one expert tensor; ``None`` dtype means BF16-like."""
        moe = self.layers[layer][0].moe_config
        hidden = moe.hidden_dim_unpadded
        intermediate = moe.intermediate_size_per_partition_unpadded * moe.moe_parallel_config.tp_size
        rows, cols = (hidden, intermediate) if projection == "down_proj" else (intermediate, hidden)
        return {
            "weight": ((rows, cols), None),
            "weight_packed": ((rows, cols // 2), torch.uint8),
            "weight_scale": ((rows, cols // 16), torch.float8_e4m3fn),
            "weight_global_scale": ((1,), torch.float32),
            "input_global_scale": ((1,), torch.float32),
        }[suffix]

    def _check_layout(self, name: str, tensor: torch.Tensor, keys: list[tuple[int, int, str, str]]) -> None:
        layer, _, projection, suffix = keys[0]
        shape, dtype = self._layout(layer, projection, suffix)
        packed = _PACKED_EXPERT_TENSOR_RE.search(name)
        if packed is not None:
            rows = shape[0] * (2 if packed.group(2) == "gate_up_proj" else 1)
            shape = (self.layers[layer][0].global_num_experts, rows, shape[1])
        if dtype is None:
            ok = tensor.dtype in (torch.bfloat16, torch.float16, torch.float32)
        else:
            ok = tensor.dtype == dtype
        if tuple(tensor.shape) != shape or not ok:
            raise RuntimeError(
                f"NVFP4_PER_TOKEN refit got {name} as {tuple(tensor.shape)}/{tensor.dtype}, "
                f"expected {shape}/{dtype or 'bf16'}"
            )

    def check_bucket(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in weights:
            keys = self._keys(name, tensor)
            if keys and keys[0] in self.expected:
                self._check_layout(name, tensor, keys)
            for key in keys:
                layer, expert, projection, suffix = key
                if key not in self.expected:
                    raise RuntimeError(f"NVFP4_PER_TOKEN refit contains unexpected expert tensor {name}")
                if key in self.seen:
                    raise RuntimeError(f"NVFP4_PER_TOKEN refit contains duplicate expert tensor {name}")
                self.seen.add(key)
                if suffix == "input_global_scale" and not torch.all(tensor == 1):
                    raise RuntimeError(f"NVFP4_PER_TOKEN activations are scaled per token, but {name} is not 1")
                if suffix == "weight_global_scale" and projection != "down_proj":
                    # The kernel applies one global scale to the fused gate/up weight.
                    other = self.gate_up_global_scales.setdefault((layer, expert), tensor.detach().float().cpu())
                    if not torch.equal(other, tensor.detach().float().cpu()):
                        raise RuntimeError(f"NVFP4_PER_TOKEN gate_proj and up_proj global scales differ at {name}")

    def finish(self) -> None:
        missing = sorted(self.expected - self.seen)
        if missing:
            raise RuntimeError(f"NVFP4_PER_TOKEN refit is missing {len(missing)} expert tensors: {missing[:8]}")


@torch.no_grad()
def _attest_scale_references(module) -> None:
    """The retained kernel must quantize per token and read the current scales from the captured storage."""
    from verl.utils.vllm.vllm_fp4_utils import _attest_nvfp4_kernel_scale_references

    experts = getattr(getattr(module.quant_method, "moe_kernel", None), "fused_experts", None)
    if not getattr(experts, "per_token_activation", False):
        raise RuntimeError("NVFP4_PER_TOKEN MoE kernel does not quantize activations per token")
    _attest_nvfp4_kernel_scale_references(module)
    actual = getattr(experts, "g1_scale_c", None)
    registered = getattr(module, "g1_scale_c", None)
    if not isinstance(actual, torch.Tensor) or not isinstance(registered, torch.Tensor):
        raise RuntimeError("NVFP4_PER_TOKEN derived scale g1_scale_c is missing")
    if actual.data_ptr() != registered.data_ptr():
        raise RuntimeError("NVFP4_PER_TOKEN kernel has a stale g1_scale_c reference")
    quant_config = experts.quant_config
    expected = quant_config.a2_gscale
    if experts.moe_config.is_act_and_mul:
        expected = quant_config.g1_alphas * quant_config.a2_gscale
    if not torch.equal(registered, expected):
        raise RuntimeError("NVFP4_PER_TOKEN g1_scale_c does not match the current weight scales")


def clear_kv_cache(model: torch.nn.Module) -> None:
    """Zero the KV cache that vLLM's start-up forwards wrote with the dummy weights.

    Dummy loading leaves the FP8 block scales at zero, so the MoE intermediate rows are
    all zero and the per-token activation quantization turns them into NaN. Attention
    reads unwritten slots of a partly filled block, so a stale NaN there poisons later
    requests unless a sleep discards the cache first.
    """
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    caches = {}
    for module in model.modules():
        # The common base of every cache-holding layer (Attention, MLAAttention, ...).
        if isinstance(module, AttentionLayerBase):
            kv = getattr(module, "kv_cache", None)
            for tensor in kv if isinstance(kv, list | tuple) else [kv]:
                if isinstance(tensor, torch.Tensor):
                    caches[tensor.data_ptr()] = tensor
    with torch.no_grad():
        for tensor in caches.values():
            tensor.zero_()


def attest_nvfp4_per_token_runtime(model: torch.nn.Module, quantization_config: dict) -> None:
    """Check the routed-expert precision of every layer and the scale state of the NVFP4 ones."""
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

    wrong = []
    for layer, (module, quantized) in sorted(routed_expert_layers(model, quantization_config).items()):
        method = module.quant_method
        if quantized and is_nvfp4_per_token_method(method):
            if not getattr(module, "_already_called_process_weights_after_loading", False):
                raise RuntimeError(f"NVFP4_PER_TOKEN MoE layer {layer} was not post-processed")
            _attest_scale_references(module)
        elif quantized or not isinstance(method, UnquantizedFusedMoEMethod):
            wrong.append((layer, type(method).__name__, "nvfp4_per_token" if quantized else "bf16"))
    if wrong:
        raise RuntimeError(
            f"NVFP4_PER_TOKEN rollout has the wrong routed-expert precision (layer, method, expected): {wrong}"
        )
