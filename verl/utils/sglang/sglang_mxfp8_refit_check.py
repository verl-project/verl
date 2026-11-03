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
"""Post-sync self-check for SGLang's MXFP8 layers.

The refit loader (``sglang_quant_utils.load_and_reprocess``) re-derives every MXFP8
layer's kernel layout on the closing request of a sync. Before that,
``snapshot_mxfp8_moe_for_check`` copies one local expert's canonical ``w13`` / ``w2``
weights and scales from the smallest MXFP8 MoE layer; after it, the smallest dense
MXFP8 layer and that expert run through their own kernels and are compared with the
dequantized reference (``verl.utils.mxfp8_refit_check``).

The expert probe makes up only the router logits (0 for the probed expert,
``_PROBE_OTHER_LOGIT`` for every other one) and runs them through the model's own
``TopK``, so each MoE runner gets the routing input it consumes: FlashInfer TRT-LLM
routes inside the kernel from the logits, the others take a precomputed top-k. The
reference reads ``w13`` in the order the engine stores it (``[up; gate]`` on the
TRT-LLM runners). The probe is skipped, with a warning once per layer, where such
logits cannot pin one expert at weight 1 (sigmoid, grouped, biased or custom routing,
fused shared experts, a routed scaling factor, or a TRT-LLM routing method other than
the softmax ones), under expert parallelism, and under TP without ``reduce_results``.

A failed probe is reported (``VERL_MXFP8_REFIT_CHECK=raise`` stops the sync). The probes
run eagerly, so storage that a CUDA graph captured is not checked.
``VERL_MXFP8_REFIT_CHECK=0`` disables them.
"""

from __future__ import annotations

import logging
import os

import torch

from verl.utils.sglang.sglang_fp8_utils import _canonical_scale_shape, _is_mxfp8_linear, _is_mxfp8_moe

logger = logging.getLogger(__name__)
# SGLang's scheduler processes fall back to the root WARNING level and would drop the INFO evidence lines.
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def self_check_mxfp8_linear(model: torch.nn.Module) -> None:
    """Probe the smallest dense MXFP8 layer after re-processing (see ``mxfp8_refit_check``)."""
    from verl.utils.mxfp8_refit_check import assert_mxfp8_linear_matches, refit_check_enabled

    if not refit_check_enabled():
        return
    candidate = None
    for name, module in model.named_modules():
        if not _is_mxfp8_linear(module) or not hasattr(module, "weight"):
            continue
        if candidate is None or module.weight.numel() < candidate[1].weight.numel():
            candidate = (name, module)
    if candidate is None:
        return
    name, module = candidate
    qm = module.quant_method
    assert_mxfp8_linear_matches(
        name,
        lambda x: qm.apply(module, x),
        module.weight.data,
        module.weight_scale_inv.data,
        engine="sglang",
    )


def snapshot_mxfp8_moe_for_check(model: torch.nn.Module):
    """Pick the smallest MXFP8 MoE layer and copy one local expert's canonical weights / scales.

    Must run after ``load_weights`` and before ``reprocess_mxfp8_moe_layers``: the
    scales are canonical ``[E, N, K/32]`` uint8 at that point (staging buffers for
    runners that rewrite them, the live buffers for runners that keep them). Returns
    ``None`` when the check is disabled, no MoE layer qualifies, or the layer's
    parallelism or routing is one the probe cannot reproduce.
    """
    from verl.utils.mxfp8_refit_check import refit_check_enabled

    if not refit_check_enabled():
        return None
    best = None
    for name, module in model.named_modules():
        if not _is_mxfp8_moe(module) or not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
            continue
        if best is None or module.w13_weight.numel() < best[1].w13_weight.numel():
            best = (name, module)
    if best is None:
        return None
    name, module = best
    if getattr(module, "moe_ep_size", 1) > 1:
        _warn_skip(
            name,
            f"expert parallelism (moe_ep_size={module.moe_ep_size}); the probe routes every row to local expert 0, "
            "which is global expert 0 only without it",
        )
        return None
    if getattr(module, "moe_tp_size", 1) > 1 and not getattr(module, "reduce_results", False):
        _warn_skip(
            name,
            f"TP {module.moe_tp_size} without reduce_results; the layer returns this rank's partial sum and the "
            "probe compares reduced outputs only",
        )
        return None
    topk = _find_topk_module(model, name)
    problem = _probe_routing_problem(module, topk)
    if problem is not None:
        _warn_skip(name, f"{problem}; the probe needs softmax routing it can send to one expert at weight 1")
        return None
    w13, w2 = module.w13_weight.data, module.w2_weight.data
    s13, s2 = module.w13_weight_scale_inv.data, module.w2_weight_scale_inv.data
    if (
        s13.dtype != torch.uint8
        or s2.dtype != torch.uint8
        or tuple(s13.shape) != _canonical_scale_shape(w13)
        or tuple(s2.shape) != _canonical_scale_shape(w2)
    ):
        _warn_skip(
            name,
            "its expert scales are not the canonical uint8 [E, N, K/32] tensors after the weight sync, so there is "
            "no reference for the kernel",
        )
        return None
    expert = 0  # local expert 0 == global expert 0 when EP is off
    return (
        name,
        module,
        expert,
        w13[expert].detach().clone(),
        s13[expert].detach().clone(),
        w2[expert].detach().clone(),
        s2[expert].detach().clone(),
        topk,
        _w13_up_first(module),
    )


# (layer, reason) pairs already reported, so a probe that cannot run on a layer warns once, not every sync.
_WARNED: set[tuple[str, str]] = set()


def _warn_skip(name: str, reason: str) -> None:
    """Warn, once per layer and reason, that the expert probe cannot run on this layer."""
    if (name, reason) not in _WARNED:
        _WARNED.add((name, reason))
        logger.warning("mxfp8 MoE refit check skipped on %s (sglang): %s", name, reason)


# Routing the probe's logits steer exactly: softmax over experts, then (Renormalize) or before (Default /
# RenormalizeNaive) the top-k. With logit 0 for the probed expert and _PROBE_OTHER_LOGIT for every other one, the
# probed expert gets weight 1.0 and the others exactly 0 in every case.
_PROBE_ROUTING_METHODS = ("Default", "Renormalize", "RenormalizeNaive")
_PROBE_OTHER_LOGIT = -1.0e4


def _find_topk_module(model: torch.nn.Module, name: str):
    """The sglang ``TopK`` that routes for the fused-MoE layer ``name``: its one sibling of that type (``mlp.topk``)."""
    from sglang.srt.layers.moe.topk import TopK

    parent = model.get_submodule(name.rpartition(".")[0])
    found = [m for m in parent.children() if isinstance(m, TopK)]
    return found[0] if len(found) == 1 else None


def _probe_routing_problem(module: torch.nn.Module, topk) -> str | None:
    """Why the probe's logits cannot route every row to one expert at weight 1 on this layer, or None if they can."""
    if topk is None:
        return "no single sglang TopK module next to the experts"
    cfg = topk.topk_config
    if getattr(cfg, "scoring_func", "softmax") != "softmax":
        return f"{cfg.scoring_func} scoring"
    if cfg.use_grouped_topk or cfg.correction_bias is not None or cfg.custom_routing_function is not None:
        return "grouped, biased or custom top-k routing"
    if cfg.num_fused_shared_experts:
        return "shared experts fused into the routed ones"
    runner_cfg = getattr(module, "moe_runner_config", None)
    for factor in (cfg.routed_scaling_factor, getattr(runner_cfg, "routed_scaling_factor", None)):
        if factor not in (None, 1.0):
            return f"routed scaling factor {factor}"
    if getattr(module, "use_flashinfer_trtllm_moe", False):
        # FlashInfer TRT-LLM routes inside the kernel by the layer's routing method; Fp8MoEMethod passes an unset
        # (or Default == 0) method as DeepSeekV3.
        from sglang.srt.layers.moe.utils import RoutingMethodType

        method = getattr(module, "routing_method_type", None)
        method = RoutingMethodType(int(method)).name if method else "DeepSeekV3"
        if method not in _PROBE_ROUTING_METHODS:
            return f"FlashInfer TRT-LLM routing method {method}"
    return None


def _w13_up_first(module: torch.nn.Module) -> bool:
    """Whether sglang stores this layer's ``w13`` as ``[up; gate]`` rather than ``[gate; up]``.

    ``FusedMoE`` swaps the w1 / w3 shard ids at load on the FlashInfer TRT-LLM runners ("Flashinfer assumes w31
    format", sglang 0.5.12 to 0.5.20) for ``Fp8MoEMethod``, and ``_load_w13`` puts up first for quant methods with
    ``load_up_proj_weight_first``; both together cancel.
    """
    return bool(getattr(module, "use_flashinfer_trtllm_moe", False)) != bool(
        getattr(module.quant_method, "load_up_proj_weight_first", False)
    )


def _route_everything_to(module: torch.nn.Module, expert: int, topk: torch.nn.Module):
    """Return ``apply_fn(x)`` running the MoE layer as the model does, with every row routed to ``expert`` at weight 1.

    Only the router logits are made up (0 for ``expert``, ``_PROBE_OTHER_LOGIT`` for every other one): the model's
    own ``TopK`` turns them into what the layer's runner consumes — the logits themselves for FlashInfer TRT-LLM,
    which routes inside the kernel, a precomputed top-k for the others — so the probe takes the routing path of
    generation. With ``top_k > 1`` other experts are selected too, at weight exactly 0.
    """

    def _apply(x: torch.Tensor) -> torch.Tensor:
        num_experts = getattr(module, "num_experts", None) or module.w13_weight.shape[0]
        logits = torch.full((x.shape[0], num_experts), _PROBE_OTHER_LOGIT, dtype=x.dtype, device=x.device)
        logits[:, expert] = 0.0
        return module.forward(x, topk(x, logits))

    return _apply


def self_check_mxfp8_moe(snapshot) -> None:
    """Probe the expert captured by ``snapshot_mxfp8_moe_for_check`` after re-processing."""
    from verl.utils.mxfp8_refit_check import assert_mxfp8_moe_expert_matches

    if snapshot is None:
        return
    name, module, expert, w13_q, s13, w2_q, s2, topk, up_first = snapshot
    reduce_ref = None
    if getattr(module, "moe_tp_size", 1) > 1:
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        reduce_ref = tensor_model_parallel_all_reduce  # the layer applies the same reduction to its output
    assert_mxfp8_moe_expert_matches(
        name,
        expert,
        _route_everything_to(module, expert, topk),
        w13_q,
        s13,
        w2_q,
        s2,
        engine="sglang",
        reduce_ref=reduce_ref,
        up_first=up_first,
    )
