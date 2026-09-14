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
"""Fail loudly when a quantized rollout is producing garbage.

Two silent failures were hit while validating MXFP8 on 2xB200, both with exit
code 0 and every metric still being logged:

- the vocabulary projection got quantized, rollout logits went ``nan``, sampling
  never emitted EOS, reward was 0.000 for the whole run;
- an engine's kernel scale layout was not rebuilt after weight sync, kl between
  trainer and rollout was 7.2 (normal: 0.001-0.03), entropy ~ log(vocab), every
  response hit the length cap.

Both are visible in step-level metrics from step 1. This sentinel reads the
metrics the trainers already compute and raises instead of logging when the
rollout is quantized (``actor_rollout_ref.rollout.quantization`` set) and any of
these hold:

1. ``training/rollout_probs_diff_mean`` is non-finite while the batch had valid tokens;
2. ``rollout_corr/kl`` is non-finite or above ``kl_max`` (default 1.0);
3. ``response_length/clip_ratio`` is ~1.0 (every response truncated at the cap) on
   ``clip_steps`` consecutive steps (default 2).

Disable with ``VERL_QUANT_SENTINEL=0``; tune with ``VERL_QUANT_SENTINEL_KL_MAX`` and
``VERL_QUANT_SENTINEL_CLIP_STEPS``. Runs without rollout quantization are never checked.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

_KEY_DIFF_MEAN = "training/rollout_probs_diff_mean"
_KEY_DIFF_VALID = "training/rollout_probs_diff_valid"
_KEY_KL = "rollout_corr/kl"
_KEY_CLIP = "response_length/clip_ratio"


def _as_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class QuantizedRolloutSentinel:
    def __init__(
        self,
        quantization: str | None,
        *,
        enabled: bool = True,
        kl_max: float = 1.0,
        clip_steps: int = 2,
    ):
        self.quantization = quantization
        self.enabled = enabled and quantization is not None
        self.kl_max = kl_max
        self.clip_steps = clip_steps
        self._clip_streak = 0

    @classmethod
    def from_config(cls, config) -> QuantizedRolloutSentinel:
        rollout_cfg = config.actor_rollout_ref.rollout
        quantization = rollout_cfg.get("quantization", None) if hasattr(rollout_cfg, "get") else None
        enabled = os.environ.get("VERL_QUANT_SENTINEL", "1") != "0"
        kl_max = float(os.environ.get("VERL_QUANT_SENTINEL_KL_MAX", "1.0"))
        clip_steps = int(os.environ.get("VERL_QUANT_SENTINEL_CLIP_STEPS", "2"))
        return cls(quantization, enabled=enabled, kl_max=kl_max, clip_steps=clip_steps)

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
        if clip is not None:
            self._clip_streak = self._clip_streak + 1 if clip >= 0.999 else 0
            if self._clip_streak >= self.clip_steps:
                problems.append(
                    f"{_KEY_CLIP} = {clip:.3f} on {self._clip_streak} consecutive steps (every response hit the "
                    "length cap, i.e. the sampler is not emitting EOS)"
                )

        if not problems:
            return
        raise RuntimeError(
            f"Quantized rollout ({self.quantization}) looks broken at step {step}: "
            + "; ".join(problems)
            + ". Likely causes: a layer that must stay in high precision was quantized (lm_head, embeddings, MoE "
            "router - check ignored_layers), the engine's kernel scale layouts were not rebuilt after the weight "
            "sync (see the MXFP8 refit loader / self-check), or an unsupported GEMM backend was selected via "
            "engine_kwargs. The run is stopped here instead of spending the remaining steps on garbage samples. "
            "Set VERL_QUANT_SENTINEL=0 to disable this check."
        )


def check_quantized_rollout_metrics(trainer, metrics: dict[str, Any], step: int) -> None:
    """Trainer hook: build the sentinel from ``trainer.config`` once, then check each step."""
    sentinel = getattr(trainer, "_quant_sentinel", None)
    if sentinel is None:
        try:
            sentinel = QuantizedRolloutSentinel.from_config(trainer.config)
        except Exception as err:  # noqa: BLE001 - never let the guard itself break training
            logger.warning("quantized-rollout sentinel disabled: could not read config (%s)", err)
            sentinel = QuantizedRolloutSentinel(None, enabled=False)
        trainer._quant_sentinel = sentinel
    sentinel.check(metrics, step)
