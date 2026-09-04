# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""In-place MXFP4 routed-expert refit for SGLang, via ``--custom-weight-loader``.

Counterpart to ``verl/utils/vllm/vllm_fp4_utils.py``, solving the same problem
one engine over: a DSv4 checkpoint ships its routed experts MXFP4-packed and the
Megatron bridge exports them in that same layout, but by the time a sync arrives
the engine's expert parameters are no longer in checkpoint layout. SGLang's
``Mxfp4FlashinferCutlassMoEMethod.process_weights_after_loading`` runs the SM90
CUTLASS interleave once at model load and publishes the result as four brand-new
``Parameter`` objects::

    layer.w13_weight           = Parameter(w13_il, requires_grad=False)
    layer.w2_weight            = Parameter(w2_il,  requires_grad=False)
    layer.w13_weight_scale_inv = Parameter(w13_s_il, requires_grad=False)
    layer.w2_weight_scale_inv  = Parameter(w2_s_il,  requires_grad=False)

Those carry none of the loader attributes ``set_weight_attrs`` had attached, so a
refit dies immediately in ``deepseek_v4.py``'s ``weight_loader = param.weight_loader``
with ``AttributeError: 'Parameter' object has no attribute 'weight_loader'``.

The cycle here is stage -> load -> replay -> fold back:

1. Swap each expert param for a checkpoint-layout buffer with the loader
   attributes reattached, preferring a byte-level reinterpret of the live
   storage so the reload lands where the kernel already reads. The interleave is
   a permutation, so the byte counts match and this costs no extra memory --
   which for 256 experts is the difference between free and doubling MoE VRAM.
2. Let ``model.load_weights`` write the refit stream into those buffers.
3. Re-run ``process_weights_after_loading`` to redo the interleave. It is a pure
   function of the checkpoint bytes (``interleave_moe_{weights,scales}_for_sm90_mixed_gemm``
   read nothing else), so replaying it reproduces the load-time layout exactly.
4. Copy its output back into the parameters the CUDA graph captured and reinstate
   those objects, so the addresses baked into the graph stay valid.

Unlike the vLLM side there is no ``replace_parameter`` seam to intercept: SGLang
assigns the four parameters directly. It does so in one place at the end of the
method, though, which makes an after-the-fact fold-back sufficient and spares us
the monkey-patch.

Why a sentinel: verl posts a full sync as a series of ~512 MB buckets, one
``update_weights_from_tensor`` each, and this loader is invoked per bucket with
no notion of position in the stream. Staging must happen once before the first
bucket and the replay once after the last, so staging is lazy (first bucket that
finds unstaged layers does it) and the sender appends ``__mxfp4_end__`` to the
final bucket to trigger the replay. Same shape as the ``__delta_spec__``
protocol in ``delta_loader.py``.

Register at server launch (verl config)::

    +actor_rollout_ref.rollout.engine_kwargs.sglang.custom_weight_loader='["verl.workers.rollout.sglang_rollout.mxfp4_loader.load_mxfp4"]'
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

LOADER_FQN = "verl.workers.rollout.sglang_rollout.mxfp4_loader.load_mxfp4"

# Zero-length marker appended to the last bucket of a sync. Carries no data; its
# presence is the whole message.
END_SENTINEL = "__mxfp4_end__"

# The four params ``process_weights_after_loading`` replaces. Bias params are not
# listed: DSv4 has no MoE bias, and a name that never existed would be skipped
# anyway.
_EXPERT_PARAMS = ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv")

# E8M0 scale per 32 fp4 weights, matching ``fp4_block_k`` in SGLang's
# ``create_fp8_moe_weight_``.
_SCALE_BLOCK_K = 32

# Where the pre-refit parameters are parked for the duration of one sync.
_LIVE_ATTR = "_verl_mxfp4_live"

# Set by ``Mxfp4FlashinferCutlassMoEMethod.process_weights_after_loading``; its
# presence is the most direct evidence that a layer went through the MXFP4
# post-processing this module has to undo and redo.
_BACKEND_ATTR = "_dsv4_mxfp4_backend"


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


def _checkpoint_layout(layer) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Shapes and dtypes ``create_weights`` handed the loader, before interleave.

    Mirrors the ``is_fp4_expert`` branch of SGLang's ``create_fp8_moe_weight_``.
    Deriving them from the layer's dimensions rather than from the live
    parameters is deliberate: the live ones are already interleaved, and for the
    scales that changes the shape, so they cannot answer what the loader expects.
    """
    e = layer.num_local_experts
    h = layer.hidden_size
    i = layer.intermediate_size_per_partition
    return {
        "w13_weight": ((e, 2 * i, h // 2), torch.int8),
        "w2_weight": ((e, h, i // 2), torch.int8),
        "w13_weight_scale_inv": ((e, 2 * i, h // _SCALE_BLOCK_K), torch.float8_e8m0fnu),
        "w2_weight_scale_inv": ((e, h, i // _SCALE_BLOCK_K), torch.float8_e8m0fnu),
    }


def _staging_data(param: torch.nn.Parameter, shape, dtype: torch.dtype) -> torch.Tensor:
    """A checkpoint-layout view of ``param``'s own storage where the bytes allow.

    The SM90 interleave permutes 4-bit values within the same buffer, so the
    interleaved parameter spans exactly the bytes the checkpoint-layout one did
    and can absorb the reload in place. Only when that does not hold is a real
    allocation needed.
    """
    data = param.data
    shape = torch.Size(shape)
    if data.dtype == dtype and data.shape == shape:
        return data
    if data.is_contiguous() and data.numel() * data.element_size() == shape.numel() * _element_size(dtype):
        return data.flatten().view(dtype).reshape(shape)
    # Zero rather than empty: a param the stream happens to skip then collapses
    # the layer's output instead of feeding the kernel freed memory.
    return torch.zeros(shape, dtype=dtype, device=data.device)


def _iter_mxfp4_layers(model):
    for module in model.modules():
        if hasattr(module, _BACKEND_ATTR) and hasattr(module, "quant_method"):
            yield module


def _stage(layer) -> None:
    """Expose checkpoint-layout buffers with the loader attributes reattached."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoeWeightScaleSupported

    live: dict[str, torch.nn.Parameter] = {}
    for name, (shape, dtype) in _checkpoint_layout(layer).items():
        param = getattr(layer, name, None)
        if not isinstance(param, torch.nn.Parameter):
            continue
        live[name] = param

        staged = torch.nn.Parameter(_staging_data(param, shape, dtype), requires_grad=False)
        # ``deepseek_v4.py`` reads ``param.weight_loader`` with no fallback; the
        # rest are read through ``getattr(..., default)`` but are carried anyway
        # so the reload behaves exactly as it did at model load.
        staged.weight_loader = layer.weight_loader
        if name.endswith("_scale_inv"):
            staged.quant_method = FusedMoeWeightScaleSupported.BLOCK.value
        for attr in ("weight_padded", "is_transposed", "output_dim"):
            if hasattr(param, attr):
                setattr(staged, attr, getattr(param, attr))
        setattr(layer, name, staged)

    setattr(layer, _LIVE_ATTR, live)


def _fold_back(layer) -> None:
    """Redo the interleave and write it into the parameters the graph captured."""
    live = getattr(layer, _LIVE_ATTR, None) or {}
    delattr(layer, _LIVE_ATTR)

    layer.quant_method.process_weights_after_loading(layer)

    for name, param in live.items():
        produced = getattr(layer, name)
        data = produced.data if isinstance(produced, torch.nn.Parameter) else produced
        if data.shape != param.shape or data.dtype != param.dtype:
            raise RuntimeError(
                f"mxfp4 refit re-derived {name} as {tuple(data.shape)}/{data.dtype}, but the live "
                f"parameter is {tuple(param.shape)}/{param.dtype}; its storage cannot be updated "
                "in place, which would leave the captured CUDA graph pointing at freed memory."
            )
        if data.data_ptr() != param.data_ptr():
            param.data.copy_(data)
        setattr(layer, name, param)


def load_mxfp4(model, named_tensors) -> None:
    """SGLang custom weight loader: refit MXFP4 experts without losing the graph.

    Called inside every TP worker process, once per bucket of a sync.
    """
    tensors = [(name, t) for name, t in named_tensors if name != END_SENTINEL]
    is_last = len(tensors) != len(named_tensors)

    layers = list(_iter_mxfp4_layers(model))
    if not layers:
        # No MXFP4 experts on this model: behave like the stock path rather than
        # failing, so the loader is safe to register unconditionally.
        model.load_weights(tensors)
        return

    unstaged = [layer for layer in layers if not hasattr(layer, _LIVE_ATTR)]
    for layer in unstaged:
        _stage(layer)
    if unstaged:
        logger.info("mxfp4 refit: staged %d MoE layers for in-place reload", len(unstaged))

    model.load_weights(tensors)

    if is_last:
        for layer in layers:
            _fold_back(layer)
        logger.info("mxfp4 refit: replayed the SM90 interleave on %d MoE layers", len(layers))
