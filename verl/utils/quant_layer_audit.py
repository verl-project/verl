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
"""Audit that the training side and the rollout side quantize the same layers.

Why. "Matched" train/rollout quantization only means something if both sides
quantize the same set of layers, and today that set is decided twice, by two
unrelated mechanisms:

- training (Megatron + TE): implicitly - a layer is quantized iff it is a TE
  linear module that ran inside ``fp8_autocast`` (decoder layers, minus
  ``first_last_layers_bf16``; never the embedding, the mcore output layer or the
  torch-implemented MoE router);
- rollout (SGLang / vLLM): explicitly - a name blacklist, ``ignored_layers``
  (``lm_head``, ``model.embed_tokens``, ``model.layers.N.mlp.gate``) plus, on the
  SGLang sync side, a set of name patterns.

Nothing checks that the two agree. They happen to for dense Qwen-style models;
they silently diverge when ``first_last_layers_bf16`` is set without the matching
rollout regex, when a model names its router differently (Mixtral's
``block_sparse_moe.gate``), when a model carries modules outside the decoder
stack (vision towers), or when an engine version changes a default. The
``lm_head`` incident in #7519 was exactly such a divergence.

What. The training side leaves a precise runtime trace: after the first
optimizer step every TE linear that quantized holds an fp8 weight workspace
(``module._fp8_workspaces``). Per decoder layer (``layer_number`` on mcore's
``TransformerLayer``) that gives "this layer's GEMMs ran in fp8". The rollout
side's decision is a predicate on HF parameter names - the very predicate the
weight sync applies (SGLang) or a reconstruction of the engine's blacklist
(vLLM). This module records the HF names streamed by one weight sync, evaluates
the predicate on each, derives the training-side expectation from the layer
index, and reports every name where the two disagree.

Runs once, at the first weight sync that happens after a training step (the
first sync, before any step, has no workspaces yet). Controlled by
``VERL_QUANT_LAYER_AUDIT``: ``warn`` (default) logs the mismatches, ``raise``
raises ``RuntimeError``, ``0`` disables.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Routers (and mcore's shared-expert gate) are plain torch linears on the training side, never TE,
# regardless of how the HF checkpoint names them.
_ROUTER_RE = re.compile(r"(?:^|\.)(gate|router|shared_expert_gate)\.weight$")
# Non-linear leaves inside a decoder layer that never see a TE GEMM.
_NON_LINEAR_RE = re.compile(r"(norm|embed|rotary|bias$)", re.IGNORECASE)


def audit_mode() -> str:
    return os.environ.get("VERL_QUANT_LAYER_AUDIT", "warn").strip().lower()


@dataclass
class TrainFp8Report:
    """Which decoder layers ran fp8 GEMMs on this rank, read from TE's workspaces."""

    seen_layers: set[int] = field(default_factory=set)
    quantized_layers: set[int] = field(default_factory=set)
    outside_layers_quantized: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.quantized_layers) or bool(self.outside_layers_quantized)


def _has_fp8_workspace(module: torch.nn.Module) -> bool:
    ws = getattr(module, "_fp8_workspaces", None)
    return isinstance(ws, dict) and len(ws) > 0


def _is_transformer_layer(module: torch.nn.Module) -> bool:
    return isinstance(getattr(module, "layer_number", None), int) and "TransformerLayer" in type(module).__name__


def collect_train_fp8_report(modules: Iterable[torch.nn.Module] | torch.nn.Module) -> TrainFp8Report:
    """Walk Megatron model chunks and record, per global decoder layer, whether any TE module quantized."""
    if isinstance(modules, torch.nn.Module):
        modules = [modules]
    report = TrainFp8Report()
    for chunk in modules:
        in_layer: set[int] = set()
        for _, layer in chunk.named_modules():
            if not _is_transformer_layer(layer):
                continue
            idx = layer.layer_number - 1  # mcore layer_number is 1-based and global across PP
            report.seen_layers.add(idx)
            for sub in layer.modules():
                in_layer.add(id(sub))
                if _has_fp8_workspace(sub):
                    report.quantized_layers.add(idx)
        for name, sub in chunk.named_modules():
            if id(sub) not in in_layer and _has_fp8_workspace(sub):
                report.outside_layers_quantized.append(name)
    return report


# transformers >= 5 saves MoE experts fused and without a ``.weight`` suffix
# (``mlp.experts.gate_up_proj`` [E, 2I, H], ``mlp.experts.down_proj`` [E, H, I]).
_FUSED_EXPERT_RE = re.compile(r"\.experts\.(gate_up_proj|down_proj)$")


def is_linear_weight_name(name: str) -> bool:
    return name.endswith(".weight") or bool(_FUSED_EXPERT_RE.search(name))


def expected_train_quantized(name: str, report: TrainFp8Report) -> bool | None:
    """Training-side expectation for one HF parameter name; ``None`` when this rank cannot judge."""
    if not is_linear_weight_name(name):
        return False
    m = _LAYER_RE.search(name)
    if m is None:
        # embeddings, lm_head, vision towers, ...: mcore quantizes nothing outside the decoder stack
        return None if report.outside_layers_quantized else False
    idx = int(m.group(1))
    if idx not in report.seen_layers:
        return None  # another pipeline stage owns this layer
    if _ROUTER_RE.search(name) or _NON_LINEAR_RE.search(name):
        return False
    return idx in report.quantized_layers


def audit_layer_sets(
    report: TrainFp8Report, hf_names: Iterable[str], rollout_quantizes: Callable[[str], bool]
) -> list[str]:
    """Return one line per HF parameter on which training and rollout disagree."""
    problems = []
    for name in hf_names:
        expected = expected_train_quantized(name, report)
        if expected is None:
            continue
        actual = bool(rollout_quantizes(name))
        if actual and not expected:
            problems.append(f"{name}: the rollout-side rule quantizes it, training ran it in high precision")
        elif expected and not actual:
            problems.append(f"{name}: training ran it in fp8, the rollout-side rule does not quantize it")
    return problems


# ---------------------------------------------------------------------------
# Rollout-side predicates
# ---------------------------------------------------------------------------


def vllm_rollout_predicate(num_hidden_layers: int, keep_high_precision: Iterable[str]) -> Callable[[str], bool]:
    """Reconstruct vLLM's decision: everything linear except the blacklist ``_apply_quantization`` sends.

    vLLM decides per module via ``quantization_config.ignored_layers`` (all ``model.layers.N.mlp.gate``
    plus ``lm_head`` / ``model.embed_tokens``) and quantizes every other linear; its sync side then
    follows the live dtype, so this predicate is the engine-side rule restated on HF names.
    """
    ignored = {f"model.layers.{i}.mlp.gate" for i in range(num_hidden_layers)} | set(keep_high_precision)

    def _pred(name: str) -> bool:
        if not name.endswith(".weight"):
            return False
        prefix = name[: -len(".weight")]
        if prefix in ignored:
            return False
        if _NON_LINEAR_RE.search(name):
            return False
        return True

    return _pred


# ---------------------------------------------------------------------------
# Worker integration
# ---------------------------------------------------------------------------


class QuantLayerAuditor:
    """Records one weight sync's HF names, then compares against the training-side fp8 trace."""

    def __init__(self, rollout_quantizes: Callable[[str], bool] | None, mode: str):
        self.rollout_quantizes = rollout_quantizes
        self.mode = mode
        self.enabled = rollout_quantizes is not None and mode not in ("0", "off", "false")
        self.done = False
        self.names: list[str] = []

    @classmethod
    def from_worker(cls, worker) -> QuantLayerAuditor:
        mode = audit_mode()
        try:
            rollout_cfg = worker.config.rollout
            quantization = rollout_cfg.get("quantization", None)
            engine = worker.actor.engine
            train_fp8 = bool(
                getattr(getattr(engine, "engine_config", None), "override_transformer_config", {}).get("fp8", None)
            )
            if quantization is None or not train_fp8:
                # bf16 rollout, or quantized rollout on a bf16 trainer: no "matched" claim to audit
                return cls(None, mode)
            hf_config = engine.model_config.hf_config
            if rollout_cfg.get("name", "") == "sglang":
                from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper, build_sglang_fp8_quant_config
                from verl.utils.sglang.sglang_mxfp8_utils import (
                    SGLangMXFP8QuantizerHelper,
                    build_sglang_mxfp8_quant_config,
                )

                if quantization == "mxfp8":
                    helper = SGLangMXFP8QuantizerHelper(build_sglang_mxfp8_quant_config(hf_config))
                else:
                    helper = SGLangFP8QuantizerHelper(build_sglang_fp8_quant_config(hf_config))
                pred = helper.should_quantize_param
            else:
                from verl.utils.mxfp8_quant import MXFP8_KEEP_HIGH_PRECISION_LAYERS

                keep = MXFP8_KEEP_HIGH_PRECISION_LAYERS if quantization == "mxfp8" else ()
                pred = vllm_rollout_predicate(hf_config.num_hidden_layers, keep)
            return cls(pred, mode)
        except Exception as err:  # noqa: BLE001 - the audit must never break weight sync
            logger.warning("quantized-layer audit disabled: %s", err)
            return cls(None, mode)

    def record(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Pass the sync generator through, remembering the HF names it yields."""
        if not self.enabled or self.done:
            yield from weights
            return
        self.names = []
        for name, tensor in weights:
            self.names.append(name)
            yield name, tensor

    def run(self, modules) -> list[str] | None:
        """After a sync: compare. Returns the mismatch list, or None if not run (disabled / no fp8 trace yet)."""
        if not self.enabled or self.done or not self.names:
            return None
        report = collect_train_fp8_report(modules)
        if not report.active:
            logger.debug("quantized-layer audit: no fp8 workspaces yet (sync before the first training step)")
            return None
        problems = audit_layer_sets(report, self.names, self.rollout_quantizes)
        self.done = True
        n_q = len(report.quantized_layers)
        n_seen = len(report.seen_layers)
        if not problems:
            logger.info(
                "quantized-layer audit: training and rollout quantize the same layers (%d of %d decoder layers "
                "on this rank in fp8, %d parameters checked)",
                n_q,
                n_seen,
                len(self.names),
            )
            return []
        msg = (
            f"quantized-layer audit: training and rollout disagree on {len(problems)} parameter(s) "
            f"({n_q} of {n_seen} decoder layers on this rank ran fp8 GEMMs):\n  - "
            + "\n  - ".join(problems[:40])
            + ("\n  - ..." if len(problems) > 40 else "")
            + "\nThe 'matched' train/rollout grid only holds for layers both sides quantize. On SGLang the rule "
            "evaluated is the sync-time one (verl/utils/fp8_utils.py): a name it does not quantize is shipped "
            "unquantized even when the engine built that layer as fp8, so the engine casts bf16 into the fp8 "
            "buffer and never receives a scale. On vLLM it is the engine blacklist. Fix with "
            "quantization_config.ignored_layers / the sync rule (rollout) or first_last_layers_bf16 / model "
            "wiring (training). VERL_QUANT_LAYER_AUDIT=raise turns this into an error, =0 silences it."
        )
        if self.mode == "raise":
            raise RuntimeError(msg)
        logger.warning(msg)
        return problems
