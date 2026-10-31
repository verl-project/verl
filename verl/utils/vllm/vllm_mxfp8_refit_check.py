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
"""Post-sync self-check for vLLM's compressed-tensors MXFP8 layers.

``process_quanted_weights_after_loading`` re-derives each staged layer's kernel
layout from the canonical tensors a weight sync wrote. Before that step, the
snapshot functions copy the canonical weight and scale of the smallest MXFP8
linear layer (``CompressedTensorsLinearMethod`` with the
``CompressedTensorsW8A8Mxfp8`` scheme) and of one expert of the smallest MXFP8
MoE layer (``CompressedTensorsW8A8Mxfp8MoEMethod``); after it, the check
functions run those layers through their own kernels and compare the output
with the dequantized reference (``verl.utils.mxfp8_refit_check``). A kernel copy
that did not follow the sync is reported (``VERL_MXFP8_REFIT_CHECK=raise`` stops the
sync) instead of serving the old weights silently.

The probes run eagerly, so storage that a CUDA graph captured is not checked.
``VERL_MXFP8_REFIT_CHECK=0`` disables both checks.
"""

import logging
import os

import torch

from verl.utils.mxfp8_refit_check import (
    MXFP8_BLOCK_SIZE,
    assert_mxfp8_linear_matches,
    assert_mxfp8_moe_expert_matches,
    refit_check_enabled,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_WARNED: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(msg, *args)


def _is_canonical_scale(weight: torch.Tensor, scale: torch.Tensor) -> bool:
    """uint8 E8M0 scale of shape ``[..., N, K // 32]`` for a ``[..., N, K]`` weight: the layout the sync writes."""
    return (
        scale.dtype == torch.uint8
        and scale.ndim == weight.ndim
        and tuple(scale.shape) == (*weight.shape[:-1], weight.shape[-1] // MXFP8_BLOCK_SIZE)
    )


def _is_mxfp8_linear(module) -> bool:
    return (
        type(getattr(module, "quant_method", None)).__name__ == "CompressedTensorsLinearMethod"
        and type(getattr(module, "scheme", None)).__name__ == "CompressedTensorsW8A8Mxfp8"
        and isinstance(getattr(module, "weight", None), torch.Tensor)
        and isinstance(getattr(module, "weight_scale", None), torch.Tensor)
    )


def snapshot_mxfp8_linear_for_check(model):
    """Copy the canonical weight and scale of the smallest MXFP8 linear layer.

    Must run after ``load_weights`` and before the staged layers are re-processed:
    at that point ``weight_scale`` is the canonical uint8 ``[N, K // 32]`` tensor
    the sync just wrote, which the kernel post-processing may then rewrite.
    """
    if not refit_check_enabled():
        return None
    found = 0
    best = None
    for name, module in model.named_modules():
        if not _is_mxfp8_linear(module):
            continue
        found += 1
        if module.weight.ndim != 2 or not _is_canonical_scale(module.weight, module.weight_scale):
            continue  # kernel layout already applied: cannot serve as the reference
        if best is None or module.weight.numel() < best[1].weight.numel():
            best = (name, module)
    if best is None:
        if found:
            _warn_once(
                "linear",
                "mxfp8 refit self-check skipped: none of the %d compressed-tensors MXFP8 linear layers holds a "
                "canonical uint8 [N, K/32] scale after the weight sync, so there is no reference for the kernel",
                found,
            )
        return None
    name, module = best
    return name, module, module.weight.data.detach().clone(), module.weight_scale.data.detach().clone()


def self_check_mxfp8_linear(snapshot) -> None:
    if snapshot is None:
        return
    name, module, qweight, scale = snapshot
    qm = module.quant_method
    assert_mxfp8_linear_matches(name, lambda x: qm.apply(module, x), qweight, scale, engine="vllm")


def _is_mxfp8_moe(module) -> bool:
    return type(getattr(module, "quant_method", None)).__name__ == "CompressedTensorsW8A8Mxfp8MoEMethod" and all(
        isinstance(getattr(module, n, None), torch.Tensor)
        for n in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    )


def _moe_parallel(module):
    """(tp_size, ep_size, dp_size) of a vLLM RoutedExperts / FusedMoE module, defaulting to 1."""
    pc = getattr(getattr(module, "moe_config", None), "moe_parallel_config", None)
    tp = int(getattr(pc, "tp_size", 1) or 1)
    ep = int(getattr(pc, "ep_size", 1) or 1)
    if getattr(pc, "use_ep", False):
        ep = max(ep, 2)
    dp = int(getattr(pc, "dp_size", 1) or 1)
    return tp, ep, dp


def snapshot_mxfp8_moe_for_check(model):
    """Copy one local expert's canonical weights and scales from the smallest MXFP8 MoE layer.

    Must run after ``load_weights`` and before the staged layers are re-processed,
    while ``w13_weight_scale`` / ``w2_weight_scale`` are the canonical
    ``[E, N, K // 32]`` uint8 tensors the sync just wrote. Returns ``None`` when the
    check is disabled, no layer qualifies, or the layer's parallelism or activation
    is one the probe cannot reproduce (expert or data parallelism, non-SiLU gating);
    a layer skipped for a reason is reported once.
    """
    if not refit_check_enabled():
        return None
    best = None
    for name, module in model.named_modules():
        if not _is_mxfp8_moe(module):
            continue
        if best is None or module.w13_weight.numel() < best[1].w13_weight.numel():
            best = (name, module)
    if best is None:
        return None
    name, module = best
    tp, ep, dp = _moe_parallel(module)
    if ep > 1 or dp > 1:
        _warn_once(
            f"moe-parallel:{name}",
            "mxfp8 MoE refit check skipped on %s (vllm): expert or data parallelism (ep_size=%d, dp_size=%d); the "
            "probe routes every row to local expert 0, which is global expert 0 only without them",
            name,
            ep,
            dp,
        )
        return None
    act = getattr(module, "activation", None)
    if act is not None and "silu" not in str(getattr(act, "value", act)).lower():
        _warn_once(
            f"moe-activation:{name}",
            "mxfp8 MoE refit check skipped on %s (vllm): activation %s; the reference implements the SiLU-gated "
            "MLP only",
            name,
            act,
        )
        return None
    problem = _monolithic_routing_problem(module)
    if problem is not None:
        _warn_once(
            f"moe-routing:{name}",
            "mxfp8 MoE refit check skipped on %s (vllm): %s, which the monolithic MoE kernel applies to the probe's "
            "router logits, so they cannot route every row to one expert at weight 1",
            name,
            problem,
        )
        return None
    w13, w2 = module.w13_weight.data, module.w2_weight.data
    s13, s2 = module.w13_weight_scale.data, module.w2_weight_scale.data
    if not (_is_canonical_scale(w13, s13) and _is_canonical_scale(w2, s2)):
        _warn_once(
            "moe",
            "mxfp8 MoE refit self-check skipped on %s: its expert scales are not the canonical uint8 "
            "[E, N, K/32] tensors after the weight sync, so there is no reference for the kernel",
            name,
        )
        return None
    expert = 0  # local expert 0 == global expert 0 without EP
    return (
        name,
        module,
        expert,
        w13[expert].detach().clone(),
        s13[expert].detach().clone(),
        w2[expert].detach().clone(),
        s2[expert].detach().clone(),
    )


def _monolithic_routing_problem(module) -> str | None:
    """Why made-up router logits cannot pin one expert at weight 1 through this layer's kernel, or None.

    A monolithic kernel (``CompressedTensorsW8A8Mxfp8MoEMethod.apply_monolithic``) routes inside the kernel and
    receives the layer's routing parameters with the logits: ``routed_scaling_factor`` scales the output, and
    grouped top-k, a score correction bias or a custom routing function change which experts the logits select.
    A modular kernel takes the hand-built top-k and never sees them.
    """
    if not getattr(getattr(module, "quant_method", None), "is_monolithic", False):
        return None
    factor = getattr(module, "routed_scaling_factor", None)
    if factor not in (None, 1, 1.0):
        return f"routed scaling factor {factor}"
    if getattr(module, "use_grouped_topk", False) or (getattr(module, "num_expert_group", None) or 1) > 1:
        return "grouped top-k routing"
    if getattr(module, "e_score_correction_bias", None) is not None:
        return "a score correction bias"
    if getattr(module, "custom_routing_function", None) is not None:
        return "a custom routing function"
    return None


def _route_everything_to_vllm_expert(module, expert: int):
    """``apply_fn(x)`` running the RoutedExperts module with every row routed to ``expert`` at weight 1.

    Calls the weight holder directly, past the MoERunner (router, shared experts, final all-reduce):
    ``forward_modular`` with a hand-built top-k for modular kernels, which also bypasses the layer's
    ``routed_scaling_factor``; ``forward_monolithic`` with a one-hot-dominant logit row for monolithic
    ones, whose kernel still applies the layer's routing parameters (see ``_monolithic_routing_problem``,
    which keeps the probe off the layers where they matter).
    """

    def _apply(x: torch.Tensor) -> torch.Tensor:
        if hasattr(module, "_ensure_moe_quant_config_init"):
            module._ensure_moe_quant_config_init()
        qm = module.quant_method
        rows = x.shape[0]
        n_experts = module.w13_weight.shape[0]
        if getattr(qm, "is_monolithic", False):
            logits = torch.full((rows, n_experts), -1e4, dtype=torch.float32, device=x.device)
            logits[:, expert] = 1e4  # softmax and sigmoid routers both give this expert weight 1, others 0
            return module.forward_monolithic(x, router_logits=logits)
        k = max(1, int(getattr(module, "top_k", 1) or 1))
        ids = [(expert + i) % n_experts for i in range(k)]  # distinct ids; only the first carries weight
        idx_dtype = getattr(qm, "topk_indices_dtype", None) or torch.int32
        topk_ids = torch.tensor([ids] * rows, dtype=idx_dtype, device=x.device)
        topk_weights = torch.zeros(rows, k, dtype=torch.float32, device=x.device)
        topk_weights[:, 0] = 1.0
        return module.forward_modular(x, topk_weights, topk_ids)

    return _apply


def self_check_mxfp8_moe(snapshot) -> None:
    if snapshot is None:
        return
    name, module, expert, w13_q, s13, w2_q, s2 = snapshot
    reduce_ref = None
    tp, _, _ = _moe_parallel(module)
    kernel = getattr(module.quant_method, "moe_kernel", None)
    if tp > 1 and kernel is not None and getattr(kernel, "output_is_reduced", lambda: False)():
        # The kernel already summed the TP shards; the reference (built from local shards) must too.
        from vllm.distributed import tensor_model_parallel_all_reduce

        reduce_ref = tensor_model_parallel_all_reduce
    # tp > 1 without an internal reduction: forward_modular returns this rank's partial sum, which is
    # exactly what the local-shard reference computes, so no reduction is applied.
    assert_mxfp8_moe_expert_matches(
        name,
        expert,
        _route_everything_to_vllm_expert(module, expert),
        w13_q,
        s13,
        w2_q,
        s2,
        engine="vllm",
        reduce_ref=reduce_ref,
    )
