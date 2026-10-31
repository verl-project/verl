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
"""Flag a quantized rollout that is producing garbage.

Two silent failures were hit while validating MXFP8 on 2xB200, both with exit
code 0 and every metric still being logged:

- the vocabulary projection got quantized, rollout logits went ``nan``, sampling
  never emitted EOS, reward was 0.000 for the whole run;
- an engine's kernel scale layout was not rebuilt after weight sync, kl between
  trainer and rollout was 7.2 (normal: 0.001-0.03), entropy ~ log(vocab), every
  response hit the length cap.

Both are visible in step-level metrics from step 1. This sentinel reads the
metrics the trainers already compute and, when the rollout is quantized, reports
each step on which any of these hold:

1. ``training/rollout_probs_diff_mean`` is non-finite while the batch had valid tokens;
2. ``rollout_corr/kl`` is non-finite or above ``kl_max`` (default 1.0);
3. ``response_length/clip_ratio`` is ~1.0 and the longest response reached ``rollout.response_length``
   (every response truncated at the cap) on ``clip_steps`` consecutive steps (default 2; ``0`` turns this
   rule off, and it is off under ``rollout.ignore_eos``, where every response runs to the cap by design).
   The V1 trainer pads responses to the batch's longest one, so a clip ratio of 1.0 alone can also mean
   "every response equally short".

The rollout counts as quantized under the rule the trainer-side weight exporter uses
(``verl.utils.qat.dispatch_rollout_weight_quantizer``): ``rollout.quantization`` sets
``linear``, ``moe`` or ``config_file``, or the model's own ``config.json`` carries a
``quantization_config`` (top level or under ``text_config``). The model config is read only
when ``actor_rollout_ref.model.path`` is a local directory. ``rollout.quantization`` is an
object even when it is off, so its presence alone says nothing.

``VERL_QUANT_SENTINEL``: ``warn`` (default) logs the findings and lets the run go on,
``raise`` stops the run with ``RuntimeError``, ``0`` disables the check. Tune with
``VERL_QUANT_SENTINEL_KL_MAX`` and ``VERL_QUANT_SENTINEL_CLIP_STEPS``. Runs without
rollout quantization are never checked.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)
# verl configures the "verl" logger only in the driver; worker processes (trainer actors, vLLM/SGLang
# workers) fall back to the root WARNING level and would drop the INFO evidence lines below.
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_KEY_DIFF_MEAN = "training/rollout_probs_diff_mean"
_KEY_DIFF_VALID = "training/rollout_probs_diff_valid"
_KEY_KL = "rollout_corr/kl"
_KEY_CLIP = "response_length/clip_ratio"
_KEY_LEN_MAX = "response_length/max"
_OFF = ("0", "off", "false")


def sentinel_mode() -> str:
    """``VERL_QUANT_SENTINEL``: ``warn`` (default), ``raise`` or ``0`` (off)."""
    return os.environ.get("VERL_QUANT_SENTINEL", "warn").strip().lower()


def _as_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _field(obj: Any, key: str) -> Any:
    """``obj[key]`` for mappings (dict, DictConfig), ``obj.key`` for dataclasses and namespaces."""
    if obj is None:
        return None
    if hasattr(obj, "get"):
        return obj.get(key, None)
    return getattr(obj, key, None)


def _native_quant_method(model_path: Any) -> str | None:
    """``quant_method`` of the model's own ``quantization_config``, read from a local ``config.json``."""
    if not isinstance(model_path, str) or not os.path.isdir(model_path):
        return None
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            hf_config = json.load(f)
    except (OSError, ValueError) as err:
        logger.debug("quantized-rollout sentinel: cannot read %s/config.json (%s)", model_path, err)
        return None
    quant = hf_config.get("quantization_config") or (hf_config.get("text_config") or {}).get("quantization_config")
    if not quant:
        return None
    return str(quant.get("quant_method") or "quantized")


def rollout_quantization_label(config: Any) -> str | None:
    """Describe the quantization the rollout runs with, or return None for a high-precision rollout."""
    ref = config.actor_rollout_ref
    quantization = _field(ref.rollout, "quantization")
    if isinstance(quantization, str):  # configs older than the structured rollout.quantization
        if quantization:
            return quantization
        quantization = None  # an empty override says nothing; the checkpoint may still be quantized
    override = [f"{key}={value}" for key in ("linear", "moe", "config_file") if (value := _field(quantization, key))]
    if override:
        return ", ".join(override)
    native = _native_quant_method(_field(_field(ref, "model"), "path"))
    return f"native {native}" if native else None


class QuantizedRolloutSentinel:
    def __init__(
        self,
        quantization: str | None,
        *,
        mode: str = "warn",
        kl_max: float = 1.0,
        clip_steps: int = 2,
        ignore_eos: bool = False,
        response_cap: float | None = None,
    ):
        self.quantization = quantization
        self.mode = mode
        self.enabled = mode not in _OFF and quantization is not None
        self.kl_max = kl_max
        # Under ignore_eos every response runs to the cap by design, so the clip rule says nothing.
        self.clip_steps = 0 if ignore_eos else clip_steps
        # rollout.response_length: a clip ratio of 1.0 counts only when the longest response reached it
        self.response_cap = response_cap
        self._clip_streak = 0

    @classmethod
    def from_config(cls, config) -> QuantizedRolloutSentinel:
        quantization = rollout_quantization_label(config)
        kl_max = float(os.environ.get("VERL_QUANT_SENTINEL_KL_MAX", "1.0"))
        clip_steps = int(os.environ.get("VERL_QUANT_SENTINEL_CLIP_STEPS", "2"))
        ignore_eos = bool(_field(config.actor_rollout_ref.rollout, "ignore_eos"))
        response_cap = _as_float(_field(config.actor_rollout_ref.rollout, "response_length"))
        return cls(
            quantization,
            mode=sentinel_mode(),
            kl_max=kl_max,
            clip_steps=clip_steps,
            ignore_eos=ignore_eos,
            response_cap=response_cap,
        )

    def check(self, metrics: dict[str, Any], step: int) -> None:
        if not self.enabled:
            return
        problems = []

        diff_mean = _as_float(metrics.get(_KEY_DIFF_MEAN))
        valid = _as_float(metrics.get(_KEY_DIFF_VALID, 1.0))
        if diff_mean is not None and (valid is None or valid >= 1.0) and not math.isfinite(diff_mean):
            problems.append(f"{_KEY_DIFF_MEAN} is {diff_mean} (rollout log-probs contain non-finite values)")

        kl = _as_float(metrics.get(_KEY_KL))
        if kl is not None and (not math.isfinite(kl) or kl > self.kl_max):
            problems.append(
                f"{_KEY_KL} = {kl:.4g} exceeds {self.kl_max} (typical quantized-rollout values are 0.001-0.03)"
            )

        clip = _as_float(metrics.get(_KEY_CLIP))
        if clip is not None and self.clip_steps > 0:
            # The V1 trainer pads responses to the batch's longest, so clip_ratio 1.0 can also mean "every
            # response equally short"; count the step only if the longest response reached the cap.
            longest = _as_float(metrics.get(_KEY_LEN_MAX))
            at_cap = self.response_cap is None or longest is None or longest >= self.response_cap
            self._clip_streak = self._clip_streak + 1 if clip >= 0.999 and at_cap else 0
            if self._clip_streak >= self.clip_steps:
                problems.append(
                    f"{_KEY_CLIP} = {clip:.3f} on {self._clip_streak} consecutive steps (every response hit the "
                    "length cap, i.e. the sampler is not emitting EOS)"
                )

        if not problems:
            return
        msg = (
            f"Quantized rollout ({self.quantization}) looks broken at step {step}: "
            + "; ".join(problems)
            + ". Likely causes: a layer that must stay in high precision was quantized (lm_head, embeddings, MoE "
            "router - check ignored_layers), the engine's kernel scale layouts were not rebuilt after the weight "
            "sync (see the post-sync self-check), or an unsupported GEMM backend was selected via engine_kwargs."
        )
        if self.mode == "raise":
            raise RuntimeError(
                msg + " The run is stopped here instead of spending the remaining steps on garbage samples. "
                "VERL_QUANT_SENTINEL=warn only logs this, =0 disables the check."
            )
        logger.warning("%s The run goes on; VERL_QUANT_SENTINEL=raise stops it here, =0 disables the check.", msg)


def check_quantized_rollout_metrics(trainer, metrics: dict[str, Any], step: int) -> None:
    """Trainer hook: build the sentinel from ``trainer.config`` once, then check each step."""
    sentinel = getattr(trainer, "_quant_sentinel", None)
    if sentinel is None:
        try:
            sentinel = QuantizedRolloutSentinel.from_config(trainer.config)
        except Exception as err:  # noqa: BLE001 - never let the guard itself break training
            logger.warning("quantized-rollout sentinel disabled: could not read config (%s)", err)
            sentinel = QuantizedRolloutSentinel(None, mode="0")
        trainer._quant_sentinel = sentinel
    sentinel.check(metrics, step)
