# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
Train-Inference Consistency Patch System for VERL (NPU Environment)

This module applies patches to ensure consistency between training (Megatron/MindSpeed)
and inference (vLLM) backends when running on Ascend NPU.

Usage:
    Set environment variable before running verl:
    export TRAIN_INFER_CONSIST=1

    The patches will be automatically applied when:
    1. TRAIN_INFER_CONSIST=1 is set
    2. Training backend is Megatron (or MindSpeed)
    3. Inference backend is vLLM
    4. Running on NPU device

Environment Variables:
    TRAIN_INFER_CONSIST: Set to "1" to enable train-inference consistency patches
    VERL_INFER_BACKEND: (Optional) Override inference backend detection (e.g., "vllm", "sglang")
    VERL_TRAIN_BACKEND: (Optional) Override training backend detection (e.g., "megatron", "fsdp")
"""

import logging
import os

logger = logging.getLogger(__name__)

TRAIN_INFER_CONSIST_ENV = "TRAIN_INFER_CONSIST"
INFER_BACKEND_ENV = "VERL_INFER_BACKEND"
TRAIN_BACKEND_ENV = "VERL_TRAIN_BACKEND"

_PATCHES_APPLIED = False
_DETECTED_TRAINING_BACKEND: str | None = None
_DETECTED_INFERENCE_BACKEND: str | None = None


def is_train_infer_consist_enabled() -> bool:
    """Check if train-inference consistency mode is enabled via environment variable."""
    return os.getenv(TRAIN_INFER_CONSIST_ENV, "0") == "1"


def get_inference_backend() -> str | None:
    """
    Get inference backend from environment variable or return None.

    Returns:
        Inference backend name (e.g., "vllm", "sglang") or None if not set
    """
    return os.getenv(INFER_BACKEND_ENV, None)


def get_training_backend() -> str | None:
    """
    Get training backend from environment variable or return None.

    Returns:
        Training backend name (e.g., "megatron", "fsdp") or None if not set
    """
    return os.getenv(TRAIN_BACKEND_ENV, None)


def apply_train_infer_consist_patches(
    training_backend: str | None = None,
    inference_backend: str | None = None,
    device: str | None = None,
    force: bool = False,
) -> bool:
    """Apply runtime patches once all required conditions are met."""
    global _PATCHES_APPLIED
    global _DETECTED_TRAINING_BACKEND
    global _DETECTED_INFERENCE_BACKEND

    if not is_train_infer_consist_enabled():
        return False

    if training_backend:
        _DETECTED_TRAINING_BACKEND = training_backend.lower()
        os.environ[TRAIN_BACKEND_ENV] = _DETECTED_TRAINING_BACKEND
    if inference_backend:
        _DETECTED_INFERENCE_BACKEND = inference_backend.lower()
        os.environ[INFER_BACKEND_ENV] = _DETECTED_INFERENCE_BACKEND

    training_backend = (
        (training_backend.lower() if training_backend else None)
        or (_DETECTED_TRAINING_BACKEND.lower()
            if _DETECTED_TRAINING_BACKEND else None)
        or (get_training_backend().lower()
            if get_training_backend() else None)
    )
    inference_backend = (
        (inference_backend.lower() if inference_backend else None)
        or (_DETECTED_INFERENCE_BACKEND.lower()
            if _DETECTED_INFERENCE_BACKEND else None)
        or (get_inference_backend().lower()
            if get_inference_backend() else None)
    )

    if not training_backend or not inference_backend:
        # Two-phase init is expected: train side and inference side may report
        # their backends at different times.
        return False

    if _PATCHES_APPLIED and not force:
        return False

    if device is None:
        try:
            from verl.utils.device import get_device_name
            device = get_device_name()
        except Exception:
            device = "npu"

    if device != "npu":
        return False

    # User requirement: megatron training + vllm inference only.
    if training_backend != "megatron" or inference_backend != "vllm":
        return False

    logger.info(
        "Applying train-infer-consistency patches "
        "(device=%s, train=%s, infer=%s).",
        device,
        training_backend,
        inference_backend,
    )
    _apply_vllm_ascend_patches()
    _apply_vllm_patches()
    _apply_megatron_patches()
    _apply_mindspeed_patches()
    _PATCHES_APPLIED = True
    logger.info("Train-infer-consistency patches applied.")
    return True


def _apply_vllm_ascend_patches() -> None:
    """Apply patches to vllm-ascend for train-inference consistency."""
    try:
        from .vllm_ascend_patch import apply_vllm_ascend_train_infer_consist_patches

        apply_vllm_ascend_train_infer_consist_patches()
        logger.debug("vllm-ascend train-inference consistency patches applied.")
    except Exception as e:
        logger.warning(f"Failed to apply vllm-ascend patches: {e}")


def apply_training_batch_invariance_patches(
    training_backend: str | None = None,
    device: str | None = None,
) -> bool:
    """
    Apply batch-invariant operator replacements for training-side workers.

    This can run before two-phase backend detection is complete so that
    Megatron training workers enable deterministic batch-invariant kernels
    as early as possible.
    """
    if not is_train_infer_consist_enabled():
        return False

    training_backend = (training_backend or get_training_backend() or "").lower()
    if training_backend != "megatron":
        return False

    if device is None:
        try:
            from verl.utils.device import get_device_name
            device = get_device_name()
        except Exception:
            device = "npu"

    if device != "npu":
        return False

    try:
        from .vllm_ascend_patch import apply_batch_invariance_runtime_patches
        apply_batch_invariance_runtime_patches()
        logger.debug("training-side batch-invariance patches applied.")
        return True
    except Exception as e:
        logger.warning("Failed to apply training-side batch-invariance patches: %s", e)
        return False


def _apply_vllm_patches() -> None:
    """Apply patches to vllm for train-inference consistency."""
    try:
        from .vllm_patch import apply_vllm_train_infer_consist_patches

        apply_vllm_train_infer_consist_patches()
        logger.debug("vllm train-inference consistency patches applied.")
    except Exception as e:
        logger.warning(f"Failed to apply vllm patches: {e}")


def _apply_megatron_patches() -> None:
    """Apply patches to megatron for train-inference consistency."""
    try:
        from .megatron_patch import apply_megatron_train_infer_consist_patches

        apply_megatron_train_infer_consist_patches()
        logger.debug("megatron train-inference consistency patches applied.")
    except Exception as e:
        logger.warning(f"Failed to apply megatron patches: {e}")


def _apply_mindspeed_patches() -> None:
    """Apply patches to mindspeed for train-inference consistency."""
    try:
        from .mindspeed_patch import apply_mindspeed_train_infer_consist_patches

        apply_mindspeed_train_infer_consist_patches()
        logger.debug("mindspeed train-inference consistency patches applied.")
    except Exception as e:
        logger.warning(f"Failed to apply mindspeed patches: {e}")


# Convenience function for quick check
def maybe_apply_patches_for_worker(
    worker_type: str,
    backend: str,
    device: str | None = None,
) -> bool:
    """
    Convenience function to apply patches from worker contexts.

    This is meant to be called from worker __init__ methods.

    Args:
        worker_type: "training" or "inference"
        backend: The backend name (e.g., "megatron", "vllm")
        device: Device type, auto-detected if None

    Returns:
        True if patches were applied
    """
    if worker_type == "training":
        return apply_train_infer_consist_patches(
            training_backend=backend,
            inference_backend=get_inference_backend(),
            device=device,
        )
    elif worker_type == "inference":
        return apply_train_infer_consist_patches(
            training_backend=get_training_backend(),
            inference_backend=backend,
            device=device,
        )
    else:
        logger.warning(f"Unknown worker_type: {worker_type}")
        return False
