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
import os
import platform
import subprocess

import torch
from packaging import version

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


def get_npu_versions() -> tuple[str, str]:
    """Get the software version and CANN toolkit version for NPU devices.

    Returns:
        tuple[str, str]: A tuple of (software_version, cann_version)

    Raises:
        RuntimeError: If unable to retrieve version information
    """
    # Check npu-smi software version
    result = subprocess.run(["npu-smi", "info", "-t", "board", "-i", "1"], capture_output=True, text=True, check=True)

    # Parse software version from output
    software_version = None
    for line in result.stdout.split("\n"):
        if "Software Version" in line:
            # Extract version from line like: "Software Version : 25.3.rc1.2"
            parts = line.split(":")
            if len(parts) > 1:
                software_version = parts[1].strip().lower()
            break

    if not software_version:
        raise RuntimeError("Could not find Software Version in npu-smi output")

    # Check CANN toolkit version
    arch = platform.machine()
    if arch not in ["arm64", "aarch64", "x86_64"]:
        raise RuntimeError(f"Unsupported architecture: {arch}")

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    cann_path = os.path.join(ascend_home, f"{arch}-linux")

    if not os.path.exists(cann_path):
        raise RuntimeError(f"CANN toolkit path does not exist: {cann_path}")

    info_file = os.path.join(cann_path, "ascend_toolkit_install.info")
    if not os.path.exists(info_file):
        raise RuntimeError(f"CANN toolkit info file does not exist: {info_file}")

    # Parse version from info file
    cann_version = None
    with open(info_file) as f:
        for line in f:
            if line.startswith("version="):
                cann_version = line.split("=", 1)[1].strip().lower()
                break

    if not cann_version:
        raise RuntimeError("Could not find version in CANN toolkit info file")

    return software_version, cann_version


def check_ipc_version_support(software_version: str, cann_version: str) -> bool:
    """Check if the given software and CANN versions support IPC.

    Compares the software version and CANN toolkit version against minimum
    required versions for IPC support:
    - Software Version should be >= 25.3.rc1
    - CANN version should be >= 8.3.rc1

    Args:
        software_version: The software version string (e.g., "25.5.0", "25.3.rc1.2", "25.5.t3.b001")
        cann_version: The CANN toolkit version string (e.g., "8.3.0", "8.3.rc1")

    Returns:
        bool: True if IPC is supported, False otherwise.

    Raises:
        RuntimeError: If version format is invalid
    """
    # For software_version like "25.3.rc1.2", "25.5.0", or "25.5.t3.b001",
    # we need to extract the base version
    # Use regex to extract version with the following rules:
    # - Standard version: 25.5.0 -> 25.5.0
    # - RC version: 25.3.rc1.2 -> 25.3.rc1
    # - t suffix version: 25.5.t3.b001 -> 25.5 (only first 2 parts if third part is lowercase t)
    # - RC version: 25.3.rc1 -> 25.3.rc1
    # For versions with more than 3 parts (e.g., 25.3.rc1.2), only match the first 3 parts
    import re

    # Match version with optional rc part or lowercase t suffix:
    # - If version has lowercase t (e.g., 25.5.t3.b001), only match first 2 parts
    # - Otherwise, match up to 3 parts (e.g., 25.5.0, 25.3.rc1.2)
    ascend_version_pattern = r"(\d+\.\d+(?=\.t))|(\d+\.\d+(?:\.(?:rc\d+|\d+))?)"
    software_match = re.match(ascend_version_pattern, software_version)
    if not software_match:
        raise RuntimeError(f"Invalid software version format: {software_version}")

    # Select the matched group (either first 2 parts or up to 3 parts)
    software_base = software_match.group(1) if software_match.group(1) else software_match.group(2)

    cann_match = re.match(ascend_version_pattern, cann_version)
    if not cann_match:
        raise RuntimeError(f"Invalid CANN version format: {cann_version}")
    else:
        # Select the matched group (either first 2 parts or up to 3 parts)
        cann_base = cann_match.group(1) if cann_match.group(1) else cann_match.group(2)

    if version.parse(software_base) >= version.parse("25.3.rc1"):
        if version.parse(cann_base) >= version.parse("8.3.rc1"):
            return True
        else:
            logger.info(f"CANN version {cann_version} is below 8.3.RC1")
    else:
        logger.info(f"Software version {software_version} is below 25.3.rc1")

    return False


def is_support_ipc() -> bool:
    """Check if the device supports IPC (Inter-Process Communication).

    For GPU devices, always returns True.
    For NPU devices, checks the software version and CANN toolkit version
    to determine if IPC is supported.

    Returns:
        bool: True if IPC is supported, False otherwise.
    """
    # If CUDA is available, it's a GPU device
    if is_cuda_available:
        return True

    # For NPU devices, check the software version and CANN toolkit version
    if is_npu_available:
        try:
            software_version, cann_version = get_npu_versions()
            return check_ipc_version_support(software_version, cann_version)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute npu-smi command: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error checking IPC support: {e}") from e

    # For other devices (CPU), return False
    return False
