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

Refits load in place, like the NVFP4 refit in ``vllm_fp4_utils``: each expert
layer gets checkpoint-layout views over the storage the CUDA graphs captured,
every bucket loads straight into them, and after the last bucket each layer is
post-processed and the results are copied back into that storage. The
post-process keeps the original kernel, whose quant config points at it.
"""

import re
from contextlib import contextmanager
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
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        convert_to_nvfp4_moe_kernel_format,
        make_nvfp4_moe_kernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
        compressed_tensors_moe_w4a4_nvfp4,
    )
    from vllm.model_executor.utils import replace_parameter

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
            layer.w13_weight = torch.nn.Parameter(layer.w13_weight_packed.data, requires_grad=False)
            delattr(layer, "w13_weight_packed")
            layer.w2_weight = torch.nn.Parameter(layer.w2_weight_packed.data, requires_grad=False)
            delattr(layer, "w2_weight_packed")

            # gate_proj and up_proj share one global scale and the static input scales
            # are 1; NVFP4PerTokenRefitCheck enforces both on every refit (the dummy
            # initial load carries neither). Per-token activation scales come from the kernel.
            w13_global_scale = layer.w13_weight_global_scale
            ones = torch.ones(layer.num_experts, device=layer.w13_weight.device, dtype=torch.float32)
            (w13, w13_scale, w13_scale_2, a13_scale, w2, w2_scale, w2_scale_2, a2_scale) = (
                convert_to_nvfp4_moe_kernel_format(
                    nvfp4_backend=self.nvfp4_backend,
                    layer=layer,
                    w13=layer.w13_weight,
                    w13_scale=layer.w13_weight_scale,
                    w13_scale_2=(1.0 / w13_global_scale[:, 0]).contiguous(),
                    a13_scale=ones,
                    w2=layer.w2_weight,
                    w2_scale=layer.w2_weight_scale,
                    w2_scale_2=1.0 / layer.w2_weight_global_scale,
                    a2_scale=ones.clone(),
                    is_act_and_mul=self.moe.is_act_and_mul,
                )
            )
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
            replace_parameter(layer, "w13_input_scale", a13_scale)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
            replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
            replace_parameter(layer, "w2_input_scale", a2_scale)

            # Post-process with a kernel built from the fresh tensors. A refit copies the
            # results into the original storage, which the retained kernel (and its CUDA
            # graphs) references.
            processing_quant_config = self.get_fused_moe_quant_config(layer)
            replace_parameter(layer, "nvfp4_a1_gscale", processing_quant_config.a1_gscale)
            replace_parameter(layer, "nvfp4_a2_gscale", processing_quant_config.a2_gscale)
            processing_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=processing_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                backend=self.nvfp4_backend,
                routing_tables=layer._expert_routing_tables(),
                per_token_activation=True,
            )
            if self.moe_kernel is None:
                self.moe_quant_config = processing_quant_config
                self.moe_kernel = processing_kernel
            processing_kernel.fused_experts.process_weights_after_loading(layer)
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
    """The retained kernel must read the current scales from the captured storage."""
    method = module.quant_method
    experts = getattr(getattr(method, "moe_kernel", None), "fused_experts", None)
    if not getattr(experts, "per_token_activation", False):
        raise RuntimeError("NVFP4_PER_TOKEN MoE kernel does not quantize activations per token")
    quant_config = experts.quant_config
    for config_name, parameter_name in (
        ("g1_alphas", "w13_weight_scale_2"),
        ("g2_alphas", "w2_weight_scale_2"),
        ("w1_scale", "w13_weight_scale"),
        ("w2_scale", "w2_weight_scale"),
        ("a1_gscale", "nvfp4_a1_gscale"),
        ("a2_gscale", "nvfp4_a2_gscale"),
    ):
        actual = getattr(quant_config, config_name, None)
        registered = getattr(module, parameter_name, None)
        if not isinstance(actual, torch.Tensor) or not isinstance(registered, torch.Tensor):
            raise RuntimeError(f"NVFP4_PER_TOKEN scale reference missing: {config_name}/{parameter_name}")
        if actual.data_ptr() != registered.data_ptr():
            raise RuntimeError(f"NVFP4_PER_TOKEN kernel has a stale scale reference: {config_name}/{parameter_name}")
    actual = getattr(experts, "g1_scale_c", None)
    registered = getattr(module, "g1_scale_c", None)
    if not isinstance(actual, torch.Tensor) or not isinstance(registered, torch.Tensor):
        raise RuntimeError("NVFP4_PER_TOKEN derived scale g1_scale_c is missing")
    if actual.data_ptr() != registered.data_ptr():
        raise RuntimeError("NVFP4_PER_TOKEN kernel has a stale g1_scale_c reference")
    expected = quant_config.a2_gscale
    if experts.moe_config.is_act_and_mul:
        expected = quant_config.g1_alphas * quant_config.a2_gscale
    if not torch.equal(registered, expected):
        raise RuntimeError("NVFP4_PER_TOKEN g1_scale_c does not match the current weight scales")


def _tensor_attrs(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The module's own parameters, buffers and plain tensor attributes."""
    tensors = {name: param for name, param in module._parameters.items() if param is not None}
    tensors.update((name, buffer) for name, buffer in module._buffers.items() if buffer is not None)
    tensors.update(
        (name, value)
        for name, value in vars(module).items()
        if isinstance(value, torch.Tensor) and not name.startswith("_")
    )
    return tensors


def _stage_per_token_moe(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Give ``load_weights`` checkpoint-layout views over the live expert storage.

    Returns the live tensors, which the CUDA graphs captured, for ``_fold_per_token_moe``.
    """
    from verl.utils.vllm.vllm_fp4_utils import _mxfp4_staging_data, _nvfp4_checkpoint_layout

    live = _tensor_attrs(module)
    for name, (shape, dtype, scale_kind, checkpoint_name) in _nvfp4_checkpoint_layout(module).items():
        param = live.get(name)
        # Global scales keep their checkpoint layout and load as they are.
        if not isinstance(param, torch.nn.Parameter) or name.endswith("global_scale"):
            continue
        data = _mxfp4_staging_data(param, shape, dtype)
        if data.data_ptr() != param.data_ptr():
            raise RuntimeError(f"NVFP4_PER_TOKEN {name} {tuple(param.shape)} cannot hold its checkpoint layout {shape}")
        staged = torch.nn.Parameter(data, requires_grad=False)
        staged.weight_loader = module.weight_loader
        if scale_kind is not None:
            staged.quant_method = scale_kind
        setattr(module, checkpoint_name, staged)
    return live


@torch.no_grad()
def _fold_per_token_moe(module: torch.nn.Module, live: dict[str, torch.Tensor]) -> None:
    """Post-process one staged layer and copy the results into its live tensors.

    ``vllm_fp4_utils`` writes back in place by patching vLLM's module-level
    ``replace_parameter`` and ``convert_to_nvfp4_moe_kernel_format``. The per-token
    method above binds both itself and derives extra tensors, so its results are
    copied back here instead. This goes away once vLLM's compressed-tensors NVFP4
    MoE supports per-token activations.
    """
    method = module.quant_method
    kernel, quant_config = method.moe_kernel, method.moe_quant_config
    type(method).process_weights_after_loading(method, module)
    current = _tensor_attrs(module)
    added = sorted(set(current) - set(live))
    if added:
        raise RuntimeError(f"NVFP4_PER_TOKEN refit derived tensors the CUDA graphs do not know: {added}")
    for name, tensor in live.items():
        new = current.get(name)
        if new is None:
            raise RuntimeError(f"NVFP4_PER_TOKEN refit dropped {name}")
        if new.data_ptr() != tensor.data_ptr():
            if new.shape != tensor.shape or new.dtype != tensor.dtype:
                raise RuntimeError(
                    f"NVFP4_PER_TOKEN refit re-derived {name} as {tuple(new.shape)}/{new.dtype}, "
                    f"but the CUDA-graph tensor is {tuple(tensor.shape)}/{tensor.dtype}"
                )
            tensor.copy_(new)
        if name in module._parameters:
            module._parameters[name] = tensor
        elif name in module._buffers:
            module._buffers[name] = tensor
        else:
            setattr(module, name, tensor)
    method.moe_kernel, method.moe_quant_config = kernel, quant_config


@contextmanager
def nvfp4_per_token_refit(model: torch.nn.Module, model_config, device: torch.device):
    """Refit in place: load every bucket inside the context, post-process on exit.

    The extra memory is one layer's post-processing. The rest of the model is
    post-processed as the BF16 refit does it; KV-cache scales are not refit.
    """
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    from verl.utils.vllm.vllm_unquant_utils import fold_unquantized_moe_params, stage_unquantized_moe_params

    staged = {
        module: _stage_per_token_moe(module)
        for module in model.modules()
        if is_nvfp4_per_token_method(getattr(module, "quant_method", None))
    }
    staged_bf16 = stage_unquantized_moe_params(model)
    yield

    def fold(layer):
        _fold_per_token_moe(layer, staged[layer])

    overridden = []
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if module in staged:
            method.process_weights_after_loading = fold
        elif isinstance(method, BaseKVCacheMethod):
            method.process_weights_after_loading = lambda layer: None
        else:
            continue
        overridden.append(method)
    try:
        with fold_unquantized_moe_params(staged_bf16):
            process_weights_after_loading(model, model_config, device)
    finally:
        for method in overridden:
            del method.process_weights_after_loading


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
