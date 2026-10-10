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

Runs once, at the first weight sync that happens after an fp8 forward (the
first sync, before any step, has no workspaces yet). Controlled by
``VERL_QUANT_LAYER_AUDIT``: ``warn`` (default) logs the mismatches, ``raise``
raises ``RuntimeError``, ``0`` disables.

Two things can remove the signal, and the audit says so instead of staying
silent. verl drops the workspaces when it offloads the training model
(``offload_megatron_model_to_cpu``, at the end of every ``train_mode()`` when
``param_offload`` is on), so the clearing helper first marks every module that
held one (``remember_fp8_workspaces``) and the audit accepts the marker. And TE
only caches the fp8 weight when mcore passes ``is_first_microbatch``, which
``disable_parameter_transpose_cache=True`` turns off; with that flag there is
nothing to read, and the audit warns once that it cannot run.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import torch


def _configure_audit_logger() -> logging.Logger:
    """Route the audit's own output to stderr, past whatever the root logger was set to.

    The audit's evidence lines are INFO/WARNING. Inside the trainer worker process a vLLM transitive
    dependency (``model_hosting_container_standards.logging_config.configure_root_logger``, pulled in
    for SageMaker) raises the root logger and every root handler to ERROR at import time; anything that
    reaches this logger and then propagates to the root handlers is silently dropped - which is why the
    audit was invisible on the first B200 runs even though it was executing. A dedicated stderr handler
    on this logger with ``propagate=False`` carries the audit's INFO/WARNING lines out on their own,
    independent of the root configuration.
    """
    log = logging.getLogger(__name__)
    log.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO").upper())
    log.propagate = False
    handler_name = "verl.quant_audit.stderr"
    if not any(getattr(h, "name", None) == handler_name for h in log.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(handler_name)
        handler.setLevel(logging.NOTSET)  # let the logger's own level decide
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [quant-audit pid=%(process)d] %(message)s"))
        log.addHandler(handler)
    return log


logger = _configure_audit_logger()

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Routers (and mcore's shared-expert gate) are plain torch linears on the training side, never TE,
# regardless of how the HF checkpoint names them.
_ROUTER_RE = re.compile(r"(?:^|\.)(gate|router|shared_expert_gate)\.weight$")
# Non-linear leaves inside a decoder layer that never see a TE GEMM.
_NON_LINEAR_RE = re.compile(r"(norm|embed|rotary|bias$)", re.IGNORECASE)


def audit_mode() -> str:
    return os.environ.get("VERL_QUANT_LAYER_AUDIT", "warn").strip().lower()


def trace(where: str, **state) -> None:
    """Log one checkpoint of the audit's own execution, at WARNING, when tracing is on.

    A guard that produces nothing is impossible to diagnose from the outside: on the first B200 runs
    the audit was silent and telling "never installed" from "installed but never reached" from "reached
    but found nothing" needed three separate instrumented runs, each patched by hand on the pod.
    ``VERL_QUANT_LAYER_AUDIT_TRACE=1`` makes every one of those states visible in a single run, at a
    level no logging configuration drops, without touching the source on the machine.
    """
    if os.environ.get("VERL_QUANT_LAYER_AUDIT_TRACE", "0") != "1":
        return
    detail = " ".join(f"{k}={v}" for k, v in state.items())
    logger.warning("quantized-layer audit trace: %s%s", where, f" | {detail}" if detail else "")


@dataclass
class TrainFp8Report:
    """Which decoder layers ran fp8 GEMMs on this rank, read from TE's workspaces."""

    seen_layers: set[int] = field(default_factory=set)
    quantized_layers: set[int] = field(default_factory=set)
    outside_layers_quantized: list[str] = field(default_factory=list)
    # mcore's TE wrappers carry config.disable_parameter_transpose_cache; True means TE never caches
    # the fp8 weight (is_first_microbatch=None), so no workspace can ever appear.
    transpose_cache_disabled: bool = False

    @property
    def active(self) -> bool:
        return bool(self.quantized_layers) or bool(self.outside_layers_quantized)


# Set on a TE module whose fp8 weight workspace verl has dropped (param offload); survives the offload.
FP8_WORKSPACE_SEEN_ATTR = "_verl_fp8_workspace_seen"


def remember_fp8_workspaces(module: torch.nn.Module) -> bool:
    """Mark ``module`` as having held an fp8 weight workspace. Call before clearing ``_fp8_workspaces``."""
    ws = getattr(module, "_fp8_workspaces", None)
    if isinstance(ws, dict) and len(ws) > 0:
        setattr(module, FP8_WORKSPACE_SEEN_ATTR, True)
        return True
    return False


def _has_fp8_workspace(module: torch.nn.Module) -> bool:
    ws = getattr(module, "_fp8_workspaces", None)
    return (isinstance(ws, dict) and len(ws) > 0) or bool(getattr(module, FP8_WORKSPACE_SEEN_ATTR, False))


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
                if getattr(sub, "disable_parameter_transpose_cache", False):
                    report.transpose_cache_disabled = True
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
    report: TrainFp8Report,
    hf_names: Iterable[str],
    rollout_quantizes: Callable[[str], bool],
    rollout_label: str = "the rollout-side rule",
) -> list[str]:
    """Return one line per HF parameter on which training and ``rollout_quantizes`` disagree."""
    problems = []
    for name in hf_names:
        expected = expected_train_quantized(name, report)
        if expected is None:
            continue
        actual = bool(rollout_quantizes(name))
        if actual and not expected:
            problems.append(f"{name}: {rollout_label} quantizes it, training ran it in high precision")
        elif expected and not actual:
            problems.append(f"{name}: training ran it in fp8, {rollout_label} does not quantize it")
    return problems


# What the rollout-side predicate stands for, per source. Said out loud in every message so a reader
# knows whether a disagreement is about the sync, the engine as configured, or the engine as built.
ROLLOUT_LABELS = {
    "sglang": "the weight-sync rule (verl/utils/fp8_utils.py exclude + include lists)",
    "vllm": "the engine blacklist as configured (quantization_config.ignored_layers)",
    "engine": "the rollout engine's live parameters (read back from the engine)",
}


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
    """Records one weight sync's HF names, then compares against the training-side fp8 trace.

    ``run`` compares the training trace against the configured rule (``rollout_quantizes``): on SGLang
    the weight-sync rule, on vLLM the engine blacklist restated on names. On vLLM the verdict is then
    re-checked against what the engine reports about its own parameters (``recheck_against_engine``,
    available where ``collective_rpc`` returns values). Which side was compared is stated in every
    message.
    """

    def __init__(self, rollout_quantizes: Callable[[str], bool] | None, mode: str, engine: str = ""):
        self.rollout_quantizes = rollout_quantizes
        self.mode = mode
        self.engine = engine
        self.enabled = rollout_quantizes is not None and mode not in ("0", "off", "false")
        self.done = False
        self.names: list[str] = []
        self.syncs_seen = 0  # distinct weight syncs the auditor wrapped; the give-up threshold counts these
        self._explained = False  # one-shot "why this guard produced nothing" warning
        self._verdict: tuple[TrainFp8Report, list[str]] | None = None  # kept for the optional engine re-check
        self._rechecked = False

    @classmethod
    def from_worker(cls, worker) -> QuantLayerAuditor:
        mode = audit_mode()
        trace("from_worker", worker=type(worker).__name__)
        try:
            rollout_cfg = worker.config.rollout
            quantization = rollout_cfg.get("quantization", None)
            engine = worker.actor.engine
            train_fp8 = bool(
                getattr(getattr(engine, "engine_config", None), "override_transformer_config", {}).get("fp8", None)
            )
            if quantization is None or not train_fp8:
                # bf16 rollout, or quantized rollout on a bf16 trainer: no "matched" claim to audit.
                # A quantized rollout that ends up unaudited is worth a warning - it is the case where
                # the audit was expected to run and silently did not (the first B200 run hit exactly this).
                (logger.warning if quantization is not None else logger.info)(
                    "quantized-layer audit: not armed, nothing to compare (rollout quantization=%r, training "
                    "fp8=%r read from %s.engine_config.override_transformer_config); the rollout runs unaudited",
                    quantization,
                    train_fp8,
                    type(engine).__name__,
                )
                return cls(None, mode)
            hf_config = engine.model_config.hf_config
            engine_name = rollout_cfg.get("name", "")
            if engine_name == "sglang":
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
            logger.info(
                "quantized-layer audit: armed (engine=%s, rollout quantization=%s, mode=%s); runs at the first "
                "weight sync after an fp8 forward",
                engine_name,
                quantization,
                mode,
            )
            return cls(pred, mode, engine=engine_name)
        except Exception as err:  # noqa: BLE001 - the audit must never break weight sync
            logger.warning("quantized-layer audit disabled: %s", err)
            return cls(None, mode)

    def record(self, weights: Iterable[tuple[str, torch.Tensor]], modules=None) -> Iterator[tuple[str, torch.Tensor]]:
        """Pass the sync stream through, remembering the HF names it yields.

        ``modules`` makes the auditor self-contained: the comparison runs as soon as the stream is
        exhausted, so wrapping the stream where the trainer *produces* it is enough. That matters
        because the weights reach the engine by more than one route (colocated worker, checkpoint
        engine, server replicas) and only the producer is common to all of them.
        """
        trace("record", enabled=self.enabled, done=self.done, with_modules=modules is not None)
        if not self.enabled or self.done:
            yield from weights
            return
        self.names = []
        self.syncs_seen += 1  # one wrapped weight stream == one sync; give-up is judged per sync, not per run() call
        for name, tensor in weights:
            self.names.append(name)
            yield name, tensor
        trace("record.stream_end", names=len(self.names), will_run=modules is not None)
        # The train-vs-rule comparison runs here, at the tail of the stream the trainer produces, because
        # that producer is the one point common to every sync route (colocated worker, checkpoint engine,
        # server replicas). It fires only once ``done`` is set, so re-consuming the stream is a no-op; the
        # engine re-check (vLLM) happens separately in the worker once the sync itself has completed.
        if modules is not None:
            self.run(modules)

    def wants_engine_recheck(self) -> bool:
        """True when a rule-based verdict stands and is worth re-checking against the live engine."""
        return self.enabled and self.done and bool(self.names) and self._verdict is not None and not self._rechecked

    def recheck_against_engine(self, engine_truth: dict[str, bool] | None) -> list[str] | None:
        """Repeat the comparison against what the engine reports about its own parameters.

        The rule-based verdict is what the audit can always produce; this says whether the engine
        actually agrees with the rule. Only a disagreement is reported - the rule verdict already
        covered the train-vs-rule half.
        """
        if not engine_truth or self._verdict is None or self._rechecked:
            return None
        self._rechecked = True
        report, names = self._verdict
        problems = audit_layer_sets(report, names, lambda n: bool(engine_truth.get(n, False)), ROLLOUT_LABELS["engine"])
        if not problems:
            logger.info(
                "quantized-layer audit: the rollout engine's live parameters agree with the training side "
                "(%d parameters re-checked against the engine itself)",
                len(names),
            )
            return []
        msg = (
            f"quantized-layer audit: training and {ROLLOUT_LABELS['engine']} disagree on {len(problems)} "
            f"parameter(s), even though the configured rule looked consistent:\n  - "
            + "\n  - ".join(problems[:40])
            + ("\n  - ..." if len(problems) > 40 else "")
            + "\nThe engine built different layers as fp8 than the trainer quantized. Fix with "
            "quantization_config.ignored_layers (rollout) or first_last_layers_bf16 / model wiring (training)."
        )
        if self.mode == "raise":
            raise RuntimeError(msg)
        logger.warning(msg)
        return problems

    def _explain_silence(self, reason: str) -> None:
        """Warn once that the audit produced no verdict. A guard that silently does nothing is the
        failure this module exists to prevent, so the reason is reported at WARNING, not INFO."""
        if self._explained:
            return
        self._explained = True
        logger.warning(
            "quantized-layer audit produced no verdict (%s). enabled=%s engine=%s names=%d. The train/rollout "
            "layer sets are unverified for this run; VERL_QUANT_LAYER_AUDIT=0 silences the audit entirely.",
            reason,
            self.enabled,
            self.engine or "?",
            len(self.names),
        )

    def run(self, modules) -> list[str] | None:
        """After a sync: compare the training trace against the configured rollout rule.

        Returns the mismatch list, or None if not run (disabled / no fp8 trace yet). The optional
        second half - comparing against what the engine reports about its own parameters - lives in
        ``recheck_against_engine``, run by the worker after the sync completes (vLLM only).
        """
        trace("run", enabled=self.enabled, done=self.done, names=len(self.names))
        if self.done:
            return None
        if not self.enabled:
            return None  # from_worker already said why, at WARNING when the rollout is quantized
        if not self.names:
            self._explain_silence("the weight sync streamed no parameter names through the auditor")
            return None
        report = collect_train_fp8_report(modules)
        if not report.active:
            # Give up only after at least two distinct syncs left no fp8 trace - counted by syncs (record()),
            # not by run() calls, so a second run() on the same sync cannot trip this prematurely.
            if report.transpose_cache_disabled or self.syncs_seen >= 2:
                # Not "too early" any more: the trainer has run at least one step and still left no trace.
                self.done = True
                cause = (
                    "disable_parameter_transpose_cache=True: TE caches the fp8 weight only when mcore passes "
                    "is_first_microbatch, which that flag turns off"
                    if report.transpose_cache_disabled
                    else "no TE module holds an fp8 weight workspace after a training step; check that "
                    "override_transformer_config.fp8 is set on the model that ran, and that nothing other "
                    "than offload_megatron_model_to_cpu clears module._fp8_workspaces"
                )
                self._explain_silence(
                    f"no training-side fp8 trace on this rank: 0 of {len(report.seen_layers)} decoder layers "
                    f"left an fp8 weight workspace. Cause: {cause}"
                )
                return None
            logger.info(
                "quantized-layer audit: no fp8 trace yet on this rank (%d decoder layers seen, %d names recorded); "
                "expected before the first training step, will retry at the next sync",
                len(report.seen_layers),
                len(self.names),
            )
            return None
        label = ROLLOUT_LABELS.get(self.engine, "the rollout-side rule")
        problems = audit_layer_sets(report, self.names, self.rollout_quantizes, rollout_label=label)
        self.done = True
        self._verdict = (report, list(self.names))
        n_q = len(report.quantized_layers)
        n_seen = len(report.seen_layers)
        if not problems:
            logger.info(
                "quantized-layer audit: training and %s quantize the same layers (%d of %d decoder layers on this "
                "rank in fp8, %d parameters checked)",
                label,
                n_q,
                n_seen,
                len(self.names),
            )
            return []
        if self.engine == "sglang":
            meaning = (
                "This compared the layers that ran fp8 GEMMs on the training side (TE fp8 weight workspaces) "
                "against the weight-sync rule, NOT against the engine's live parameters. A name the rule does "
                "not quantize is shipped as bf16; if the engine built that layer as fp8 the loader's "
                "sync-vs-engine dtype check reports it at load time."
            )
        elif self.engine == "vllm":
            meaning = (
                "This compared the layers that ran fp8 GEMMs on the training side (TE fp8 weight workspaces) "
                "against the engine blacklist as configured, NOT against the engine's live parameters (the "
                "engine could not be asked). On vLLM the sync follows the engine's live dtype, so a "
                "disagreement means engine and trainer quantize different layers, not that the sync is broken."
            )
        else:
            meaning = (
                "This compared the layers that ran fp8 GEMMs on the training side (TE fp8 weight workspaces) "
                "against the configured rollout-side rule, NOT against the engine's live parameters."
            )
        msg = (
            f"quantized-layer audit: training and {label} disagree on {len(problems)} parameter(s) "
            f"({n_q} of {n_seen} decoder layers on this rank ran fp8 GEMMs):\n  - "
            + "\n  - ".join(problems[:40])
            + ("\n  - ..." if len(problems) > 40 else "")
            + "\n"
            + meaning
            + " Fix with quantization_config.ignored_layers / the sync rule (rollout) or first_last_layers_bf16 / "
            "model wiring (training). VERL_QUANT_LAYER_AUDIT=raise turns this into an error, =0 silences it."
        )
        if self.mode == "raise":
            raise RuntimeError(msg)
        logger.warning(msg)
        return problems
