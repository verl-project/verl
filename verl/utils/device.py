# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available(check_device=True) -> bool:
    """Check if Ascend NPU is available for PyTorch operations.

    Attempts to detect NPU availability by checking for the torch.npu module
    and its is_available() function.

    Args:
        check_device : only check torch_npu package or strictly check if NPU device is available

    Returns:
        bool: True if NPU is available, False otherwise.
    """
    try:
        if not hasattr(torch, "npu"):
            return False

        if check_device:
            return torch.npu.is_available()
        else:
            return True
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def get_resource_name() -> str:
    """Function that return ray resource name based on the device type.
    Returns:
        ray resource name string, either "GPU" or "NPU".
    """
    return "GPU" if is_cuda_available else "NPU"


def get_visible_devices_keyword() -> str:
    """Get the environment variable name for visible device selection.

    Returns the appropriate environment variable name based on the available
    accelerator type (CUDA or Ascend NPU).

    Returns:
        str: 'CUDA_VISIBLE_DEVICES' if CUDA is available,
            'ASCEND_RT_VISIBLE_DEVICES' otherwise.
    """
    return "CUDA_VISIBLE_DEVICES" if not is_torch_npu_available(check_device=False) else "ASCEND_RT_VISIBLE_DEVICES"


def get_device_name() -> str:
    """Get the device type string based on available accelerators.

    Detects the available accelerator and returns the corresponding PyTorch
    device type string. Currently supports CUDA, Ascend NPU, and CPU.

    Returns:
        str: Device type string ('cuda', 'npu', or 'cpu').
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device():
    """Get the PyTorch device module for the current accelerator.

    Returns the torch device namespace (e.g., torch.cuda, torch.npu) based on
    the detected accelerator type. Falls back to torch.cuda if the namespace
    is not found.

    Returns:
        module: The PyTorch device module (torch.cuda, torch.npu, etc.).
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """Get the index of the current accelerator device.

    Returns:
        int: The current device index (e.g., 0 for 'cuda:0').
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Get the distributed communication backend based on device type.

    Returns the appropriate collective communication backend for the
    detected accelerator (HCCL for Ascend NPU, NCCL for CUDA).

    Returns:
        str: Backend name ('hccl' for NPU, 'nccl' for CUDA/default).
    """
    if is_npu_available:
        return "hccl"
    else:
        # default to nccl
        return "nccl"


def set_expandable_segments(enable: bool) -> None:
    """Configure CUDA memory allocator expandable segments setting.

    Expandable segments can help avoid out-of-memory (OOM) errors by allowing
    the memory allocator to expand existing memory segments rather than
    allocating new ones.

    Args:
        enable: If True, enable expandable segments. If False, disable them.

    Note:
        This function only has an effect when CUDA is available.
    """
    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")


def auto_set_device(config) -> None:
    """Automatically configure device name for different accelerators.

    For example, on Ascend NPU, this function defaults the trainer device to "npu"
    unless explicitly set to "cpu".

    Args:
        config: Configuration object with trainer.device attribute.
    """
    if config and hasattr(config, "trainer") and hasattr(config.trainer, "device"):
        if is_torch_npu_available():
            if config.trainer.device not in ["cpu", "npu"]:
                logger.warning(
                    f"Detect setting config.trainer.device to {config.trainer.device} for Ascend NPU, maybe"
                    f"from default value in config file, automatically set to `npu` instead."
                )

            config.trainer.device = "npu"
        # Other cases: set device to "cuda" via config file, no need to change.


def get_device_capability(device_id: int = 0) -> tuple[int | None, int | None]:
    """Get the compute capability of a CUDA device.

    Args:
        device_id: The CUDA device index to query. Defaults to 0.

    Returns:
        tuple: A tuple of (major, minor) compute capability version,
            or (None, None) if CUDA is not available.
    """
    major, minor = None, None
    if is_cuda_available:
        major, minor = torch.cuda.get_device_capability(device_id)

    return major, minor
