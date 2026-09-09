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
"""MXFP8 weight-sync loader for SGLang, loaded via ``--custom-weight-loader``.

Why this exists. SGLang's FlashInfer MXFP8 GEMM backends do not consume the
canonical ``weight_scale_inv`` (UE8M0, ``[n, k/32]``) directly: at load time
``Fp8LinearMethod._process_mxfp8_linear_weight_scale`` derives a kernel layout
from it — ``flashinfer_cutlass`` stores an interleaved copy in
``weight_scale_inv_swizzled`` and ``apply()`` reads only that copy. A weight
sync overwrites ``weight`` / ``weight_scale_inv`` through ``model.load_weights``
and nothing re-derives the copy, so after the first sync the kernel pairs fresh
weights with stale scales and generation is garbage (observed on 2xB200,
SGLang 0.5.12: held-out acc 0.0, kl 7.2, entropy ~ log(vocab), from step 0).

What it does. ``load_and_reprocess`` is registered as a custom loader and named
as the request ``load_format``; SGLang ``dynamic_import``s it inside every TP
worker and calls it with ``(model, named_tensors)``. It loads the tensors the
standard way, then re-runs ``process_weights_after_loading`` on every module
whose quant method is MXFP8, which rebuilds the swizzled scales from the just
written canonical ones. For the default Triton backend the re-run is a no-op
(``apply()`` reads ``weight_scale_inv`` directly), so the loader is safe to use
unconditionally. The ``flashinfer_trtllm`` backend is rejected: it shuffles the
*weight* tensor itself in place at load, so re-running the processing on an
already-shuffled layer would shuffle twice; supporting it needs per-layer
tracking of which parameters a sync touched.

Register at server launch (verl does this when ``rollout.quantization=mxfp8``)::

    custom_weight_loader=["verl.workers.rollout.sglang_rollout.mxfp8_refit_loader.load_and_reprocess"]

and send every ``update_weights_from_tensor`` request with
``load_format`` set to the same import path.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)

# Import path callers pass as both --custom-weight-loader and load_format.
LOADER_FQN = "verl.workers.rollout.sglang_rollout.mxfp8_refit_loader.load_and_reprocess"


def _is_mxfp8_linear(module: torch.nn.Module) -> bool:
    qm = getattr(module, "quant_method", None)
    return (
        qm is not None
        and getattr(qm, "use_mxfp8", False)
        and hasattr(qm, "process_weights_after_loading")
        and hasattr(module, "weight_scale_inv")
    )


def reprocess_mxfp8_layers(model: torch.nn.Module) -> int:
    """Re-derive backend-specific MXFP8 scale layouts from the canonical scales.

    Returns the number of modules reprocessed. Raises for backends whose
    post-load processing is not idempotent (``flashinfer_trtllm``).
    """
    from sglang.srt.layers.quantization.fp8_utils import get_fp8_gemm_runner_backend

    backend = get_fp8_gemm_runner_backend()
    if backend.is_flashinfer_trtllm():
        raise NotImplementedError(
            "MXFP8 weight sync with fp8_gemm_runner_backend=flashinfer_trtllm is not supported: "
            "that backend shuffles the weight tensor in place at load time, so post-load processing "
            "cannot be re-run after a sync. Use flashinfer_cutlass or the default (triton) backend."
        )
    if not backend.is_flashinfer_cutlass():
        # Triton (default) consumes canonical UE8M0 scales directly; nothing to rebuild.
        return 0

    count = 0
    for name, module in model.named_modules():
        if not _is_mxfp8_linear(module):
            continue
        qm = module.quant_method
        if not getattr(qm, "is_checkpoint_fp8_serialized", True):
            # Would re-quantize from a bf16 master weight that a sync just replaced with fp8 data.
            raise RuntimeError(
                f"{name}: MXFP8 quant method expects a non-fp8-serialized checkpoint; refusing to "
                "re-quantize after a weight sync."
            )
        qm.process_weights_after_loading(module)
        count += 1
    return count


def load_and_reprocess(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Standard ``load_weights`` followed by MXFP8 scale re-processing."""
    model.load_weights(named_tensors)
    n = reprocess_mxfp8_layers(model)
    if n:
        logger.debug("mxfp8 refit: re-derived swizzled scales for %d modules", n)
