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
"""Export vs engine: is each synced weight sent in the format of the engine parameter that receives it?

This is the rollout half of the quantized-layer audit; ``verl.utils.quant_layer_audit`` holds the training
half (training vs export). Each rollout engine runs it in its own processes, inside the loader that writes a
weight sync, so it reads the very parameters the sync is about to write:

- vLLM: the colocated worker extension (``verl.workers.rollout.vllm_rollout.utils``), with the name
  resolution of ``verl.utils.vllm.vllm_sync_format_check``;
- SGLang: the MXFP8/MXFP4/NVFP4 refit loader (``verl.utils.sglang.sglang_quant_utils.load_and_reprocess``),
  with the name resolution of ``verl.utils.sglang.sglang_sync_format_check``.

It compares quantized against high precision, not exact dtypes. A quantized weight (FP8, packed FP4) sent to a
BF16 parameter, or a BF16 weight sent to a quantized one, would be cast into a buffer of the other format, and a
quantized layer would never receive its scale. The trainer-side exporter and the engine read the same
quantization config (``rollout.quantization``, or the checkpoint's own ``quantization_config``), so a
disagreement means the two matched a module differently, for example an ``ignore`` entry one side reads as a
prefix and the other as an exact name.

What a sync sends follows that config, and the engine keeps the formats it was built with, so a later sync cannot
disagree where the first one agreed. The check therefore runs on every bucket of the first weight sync, before
the bucket is written, and reports its totals once when that sync finishes; a sync that does not finish is
checked again in full.

``VERL_QUANT_LAYER_AUDIT``, shared with the training half: ``warn`` (default) logs the disagreements and lets the
sync go on, ``raise`` stops the sync before the bucket is written, ``0`` disables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import torch

from verl.utils.quant_layer_audit import audit_mode, logger

# HF leaf names that can be linear weights: ``.weight``, compressed-tensors' packed FP4 ``.weight_packed``, and
# the fused expert tensors transformers >= 5 writes without a suffix (``mlp.experts.gate_up_proj``).
_CHECKED_LEAVES = ("weight", "weight_packed", "gate_up_proj", "down_proj")
_HIGH_PRECISION = (torch.bfloat16, torch.float16, torch.float32)
_OFF = ("0", "off", "false")
_MAX_LISTED = 40

# (engine model, HF weight name) -> the engine parameter that receives it, or None when the name does not resolve
ResolveParam = Callable[[torch.nn.Module, str], Optional[torch.Tensor]]


def _format(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


class SyncFormatCheck:
    """Export vs engine for one engine model. It lives on the model, so each engine process keeps its own."""

    _ATTR = "_verl_sync_format_check"

    def __init__(self, engine: str, resolve_param: ResolveParam):
        self.engine = engine
        self.resolve_param = resolve_param
        self.finished = False  # the first weight sync has been checked end to end
        # totals of the sync being checked
        self.checked = 0  # weights compared with the parameter that receives them
        self.disagreeing = 0  # of those, sent in another format than the parameter's
        self.unmapped = 0  # names that resolve onto no engine parameter: not judged
        self._buckets = 0

    @classmethod
    def of(cls, model: torch.nn.Module, engine: str, resolve_param: ResolveParam) -> SyncFormatCheck:
        """The check attached to ``model``, created on first use."""
        check = model.__dict__.get(cls._ATTR)
        if check is None:
            check = cls(engine, resolve_param)
            model.__dict__[cls._ATTR] = check
        return check

    @property
    def pending(self) -> bool:
        return not self.finished and audit_mode() not in _OFF

    def begin_sync(self) -> None:
        """Start a weight sync. Totals left by a sync that did not finish are dropped: it is checked again in full."""
        if self.finished:
            return
        self.checked = self.disagreeing = self.unmapped = self._buckets = 0

    def check_bucket(self, model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Report the bucket's weights sent in another format than the receiving parameter's. Call before writing."""
        if not self.pending:
            return
        problems = []
        for name, tensor in named_tensors:
            if name.rsplit(".", 1)[-1] not in _CHECKED_LEAVES:
                continue
            target = self.resolve_param(model, name)
            if target is None:
                self.unmapped += 1
                continue
            self.checked += 1
            if (tensor.dtype in _HIGH_PRECISION) != (target.dtype in _HIGH_PRECISION):
                problems.append(
                    f"{name}: the sync sends {_format(tensor.dtype)}, the engine parameter is {_format(target.dtype)}"
                )
        self._buckets += 1
        self.disagreeing += len(problems)
        if problems:
            self._report(problems)

    def end_sync(self) -> None:
        """After the first finished sync: report the totals once and stop checking."""
        if not self.pending or self._buckets == 0:
            return
        self.finished = True
        logger.info(
            "quantized-layer audit, export vs %s engine, first weight sync: %d weights checked, %d in a format other "
            "than the receiving parameter's, %d names not mapped onto an engine parameter (not judged); later syncs "
            "send the same formats and are not checked again",
            self.engine,
            self.checked,
            self.disagreeing,
            self.unmapped,
        )

    def _report(self, problems: list[str]) -> None:
        msg = (
            f"quantized-layer audit: the weight sync and the {self.engine} engine disagree on the format of "
            f"{len(problems)} weight(s):\n  - "
            + "\n  - ".join(problems[:_MAX_LISTED])
            + (f"\n  - ... and {len(problems) - _MAX_LISTED} more" if len(problems) > _MAX_LISTED else "")
            + "\nThe trainer-side exporter and the engine read the same quantization config (rollout.quantization, "
            "or the checkpoint's own quantization_config) but matched these modules differently. Loading would cast "
            "each weight into a parameter of the other format, and a quantized layer would never receive its scale. "
            "Check rollout.quantization.ignore against docs/low_precision/mxfp8.md (Checks)."
        )
        if audit_mode() == "raise":
            raise RuntimeError(msg + " The sync is stopped before this bucket is written.")
        logger.warning("%s The sync goes on; VERL_QUANT_LAYER_AUDIT=raise stops it here, =0 disables the check.", msg)
