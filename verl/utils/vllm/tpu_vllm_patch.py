# Copyright 2024-2025 BAAI and Google LLC
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
"""TPU-specific vLLM patches for rotary positional embeddings."""

import importlib.util
import logging
import os

import torch

logger = logging.getLogger(__name__)


def _tpu_sign(x: torch.Tensor) -> torch.Tensor:
    """Builds [[-1], [1]] with the dtype/device of ``x``.

    This is deliberately built from tensor ops on every call instead of being
    memoized in a module-level dict. torch.compile specializes reads of Python
    globals into the serialized AOT prologue, so a cached constant resurfaces as
    a ``KeyError`` when the compiled artifact is replayed in a fresh process.
    XLA constant-folds this away, so there is no runtime cost.
    """
    return (torch.arange(2, dtype=x.dtype, device=x.device) * 2.0 - 1.0).unsqueeze(-1)


def _tpu_rotate_neox(x: torch.Tensor) -> torch.Tensor:
    """cat((-x2, x1), -1) without a concat. Bitwise identical under IEEE754."""
    half = x.shape[-1] // 2
    swapped = x.unflatten(-1, (2, half)).flip(-2)
    return (swapped * _tpu_sign(x)).flatten(-2)


def _tpu_widen(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[..., half] -> [..., 1, 2 * half], i.e. cat((t, t), -1).unsqueeze(-2)."""
    lead = t.shape[:-1]
    half = t.shape[-1]
    return t.unsqueeze(-2).expand(*lead, 2, half).reshape(*lead, 2 * half).unsqueeze(-2).to(dtype)


# TODO: Remove this workaround once upstream vLLM PR #56879 is merged and released.
# https://github.com/vllm-project/vllm/pull/56879
def patch_tpu_rotary_emb():
    """Patches vLLM ApplyRotaryEmb and rotate_neox with concat-free implementations.

    On Google TPU, torch.cat along the last dimension lowers to a strided DynamicUpdateSlice (DUS).
    When fused with elementwise ops, it fails the unaligned DUS predicate inside the XLA:TPU
    fusion emitter (b/501165531) and causes a fatal crash:
        Check failed: fusion_util::IsFusibleUnalignedDUS(...)

    This patch replaces the concatenation with an unflatten + flip formulation that is
    bitwise identical and emits no DUS instructions.
    """
    try:
        import vllm.model_executor.layers.rotary_embedding.common as rotary_common
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        if getattr(ApplyRotaryEmb, "_verl_tpu_rotary_patched", False):
            print("[ROPEPATCH] already patched", flush=True)
            return

        def patched_forward_static(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            is_neox_style: bool = True,
            enable_fp32_compute: bool = False,
        ) -> torch.Tensor:
            origin_dtype = x.dtype
            if enable_fp32_compute:
                x = x.float()

            if is_neox_style:
                cos_f = _tpu_widen(cos, x.dtype)
                sin_f = _tpu_widen(sin, x.dtype)
                output = x * cos_f + _tpu_rotate_neox(x) * sin_f
            else:
                cos = cos.unsqueeze(-2).to(x.dtype)
                sin = sin.unsqueeze(-2).to(x.dtype)
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
                o1 = x1 * cos - x2 * sin
                o2 = x2 * cos + x1 * sin
                output = torch.stack((o1, o2), dim=-1).flatten(-2)

            if enable_fp32_compute:
                output = output.to(origin_dtype)
            return output

        ApplyRotaryEmb.forward_static = staticmethod(patched_forward_static)
        rotary_common.rotate_neox = _tpu_rotate_neox
        ApplyRotaryEmb._verl_tpu_rotary_patched = True
        logger.info("Successfully applied TPU concat-free RoPE patch to vLLM.")
        print(f"[ROPEPATCH] applied pid={os.getpid()}", flush=True)
    except Exception as e:
        logger.warning(f"Failed to apply TPU rotary embedding patch to vLLM: {e}")
        print(f"[ROPEPATCH] FAILED pid={os.getpid()} err={e!r}", flush=True)


def _tpu_runtime_present() -> bool:
    """True when this process runs against the TPU backend.

    ``VERL_PLATFORM`` is the cheap signal, but it does not reach every process
    that matters. The vLLM TPU executor reuses *pooled* Ray workers —
    ``vllm_torchtpu/executors/ray_distributed_executor.py`` creates
    ``RayWorkerWrapper`` with no ``runtime_env`` — so those workers were started
    before the job existed and see ``VERL_PLATFORM`` unset, even though they are
    precisely the processes that compile and execute the model.

    Fall back to detecting the TPU backend package, which is absent on
    GPU/CPU installs.
    """
    if os.environ.get("VERL_PLATFORM") == "tpu":
        return True
    return importlib.util.find_spec("torch_tpu") is not None


def apply_tpu_vllm_patches() -> None:
    """Apply TPU-specific vLLM patches."""
    if not _tpu_runtime_present():
        return

    # Disable vLLM's AOT compile cache. The artifact this stack writes reloads as
    # "num_artifacts=0 num_submods=0", a degenerate callable that silently drops
    # the model back to eager execution, and the eager torch.cat below then trips
    # the unaligned-DUS CHECK in the XLA:TPU fusion emitter (b/501165531).
    # Measured on Qwen3-0.6B / v6e TP=8: ~85 failures warm, zero cold. Salting the
    # cache key would not help, since the bad artifact comes from a healthy run.
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    patch_tpu_rotary_emb()
