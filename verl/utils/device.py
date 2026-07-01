# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

"""Backward-compatible device utilities.

All public names in this module are preserved for existing callers (80+ import
sites).  Internally every function now delegates to the platform abstraction
layer in :mod:`verl.plugin.platform`.
"""

import logging
import os
import platform
import subprocess

import torch
from packaging import version

from verl.plugin.platform import get_platform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level availability flags (kept for backward compatibility)
# ---------------------------------------------------------------------------


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
        ray resource name string, e.g. "GPU", "NPU".
    """
    return get_platform().ray_resource_name()


def get_vendor() -> str:
    """Return the hardware vendor name for the current platform.

    Returns:
        Vendor name string, e.g. "nvidia", "metax", "huawei", "intel".
    """
    return get_platform().vendor_name


# ---------------------------------------------------------------------------
# Device info helpers
# ---------------------------------------------------------------------------


def get_visible_devices_keyword() -> str:
    """Get the environment variable name for visible device selection.

    Returns:
        str: e.g. 'CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES'.
    """
    return get_platform().visible_devices_envvar()


def get_device_name() -> str:
    """Get the device type string based on available accelerators.

    Returns:
        str: Device type string ('cuda', 'npu', 'cpu', …).
    """
    return get_platform().device_name


def get_torch_device():
    """Get the PyTorch device module for the current accelerator.

    Returns:
        module: The PyTorch device module (torch.cuda, torch.npu, etc.).
    """
    return get_platform().device_module


def get_device_id() -> int:
    """Get the index of the current accelerator device.

    Returns:
        int: The current device index (e.g., 0 for 'cuda:0').
    """
    return get_platform().current_device()


def get_nccl_backend() -> str:
    """Get the distributed communication backend based on device type.

    Returns:
        str: Backend name ('nccl', 'hccl', 'gloo', …).
    """
    return get_platform().communication_backend_name()


# ---------------------------------------------------------------------------
# Memory / allocator
# ---------------------------------------------------------------------------


def set_expandable_segments(enable: bool) -> None:
    """Configure memory allocator expandable segments setting.

    Args:
        enable: If True, enable expandable segments. If False, disable them.
    """
    get_platform().set_allocator_settings(f"expandable_segments:{enable}")


# ---------------------------------------------------------------------------
# Device auto-configuration
# ---------------------------------------------------------------------------


def auto_set_device(config) -> None:
    """Automatically configure device name for different accelerators.

    Args:
        config: Configuration object with trainer.device attribute.
    """
    if config and hasattr(config, "trainer") and hasattr(config.trainer, "device"):
        detected = get_platform().device_name
        # Only override when the config value doesn't match the detected platform
        if detected != "cpu" and config.trainer.device != detected:
            if config.trainer.device != "cpu":
                logger.warning(
                    f"Detect setting config.trainer.device to {config.trainer.device} for "
                    f"{detected}, automatically set to `{detected}` instead."
                )
            config.trainer.device = detected


# ---------------------------------------------------------------------------
# Device properties
# ---------------------------------------------------------------------------


def get_device_capability(device_id: int = 0) -> tuple[int | None, int | None]:
    """Get the compute capability of the current accelerator device.

    Args:
        device_id: The device index to query. Defaults to 0.

    Returns:
        tuple: (major, minor) or (None, None) if not applicable.
    """
    return get_platform().get_device_capability(device_id)


def get_npu_versions() -> tuple[str, str]:
    """Get the software version and CANN toolkit version for NPU devices.

    Returns:
        tuple[str, str]: A tuple of (software_version, cann_version)

    Raises:
        RuntimeError: If unable to retrieve version information
    """
    # Check npu-smi software version
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "board", "-i", "1"], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError:
        # Card 1 not found (common in K8s with non-consecutive device IDs)
        # Try first device from ASCEND_VISIBLE_DEVICES env var
        visible_devices = os.environ.get("ASCEND_VISIBLE_DEVICES")
        if not visible_devices:
            raise  # Re-raise original error if env var not set

        try:
            npu_id = int(visible_devices.split(",")[0])
        except (ValueError, IndexError):
            raise  # Re-raise original error if env var format invalid

        # Retry with the first available device from K8s
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "board", "-i", str(npu_id)], capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError:
            # On A3 machines with one-card-two-die, the device ID is a die index.
            # Try using the physical card index (npu_id // 2) instead.
            physical_card_id = npu_id // 2
            result = subprocess.run(
                ["npu-smi", "info", "-t", "board", "-i", str(physical_card_id)],
                capture_output=True,
                text=True,
                check=True,
            )

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

    Delegates to the platform abstraction layer.

    Returns:
        bool: True if IPC is supported, False otherwise.
    """
    return get_platform().is_ipc_supported()


def is_device_available() -> bool:
    """Check if any accelerator device is available.

    Returns:
        bool: True if any accelerator is available.
    """
    return get_platform().is_available()


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------


def manual_seed(seed: int) -> None:
    """Set the seed for the current accelerator device.

    Args:
        seed: The desired seed.
    """
    get_platform().manual_seed(seed)


def manual_seed_all(seed: int) -> None:
    """Set the seed for all accelerator devices.

    Args:
        seed: The desired seed.
    """
    get_platform().manual_seed_all(seed)
