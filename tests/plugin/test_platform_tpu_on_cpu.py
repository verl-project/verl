# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""The TPU platform must be describable without a TPU attached.

``PlatformTPU`` is registered long before any engine can run on it, so these tests check that
the module loads where no TPU runtime exists and that the platform answers the questions
shared verl code asks it. They need no TPU and no ``torch_tpu`` install.
"""

import sys

import pytest
import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry
from verl.plugin.platform.platform_tpu import (
    HBM_BYTES_TPU_DEFAULT,
    HBM_BYTES_TPU_V5P,
    HBM_BYTES_TPU_V6E,
    DummyTpuDeviceModule,
    PlatformTPU,
    TPUDeviceModuleProxy,
    _ensure_torch_tpu,
    get_tpu_chip_hbm_bytes,
)


@pytest.fixture()
def platform():
    return PlatformTPU()


# ---------------------------------------------------------------------------
# Loading without a TPU runtime
# ---------------------------------------------------------------------------


def test_importing_the_module_survives_a_missing_runtime():
    """platform_tpu is imported at start-up, so a missing torch_tpu must not raise."""
    assert "verl.plugin.platform.platform_tpu" in sys.modules


def test_ensure_torch_tpu_reports_false_without_the_runtime():
    if hasattr(torch, "tpu"):
        pytest.skip("torch_tpu is installed in this environment")
    assert _ensure_torch_tpu() is False


def test_registered_under_tpu():
    assert PlatformRegistry.get("tpu") is PlatformTPU


def test_constructing_the_platform_needs_no_runtime(platform):
    assert platform is not None


# ---------------------------------------------------------------------------
# Platform identity
# ---------------------------------------------------------------------------


def test_identity(platform):
    assert platform.device_name == "tpu"
    assert platform.vendor_name == "google"
    assert platform.ray_resource_name() == "TPU"
    assert platform.communication_backend_name() == "tpu_dist"


def test_ray_resource_options_uses_custom_resource(platform):
    """TPU chips are a custom Ray resource, not ``num_gpus``."""
    assert platform.ray_resource_options(4) == {"resources": {"TPU": 4}}
    assert platform.ray_resource_options(0) == {}


def test_ray_noset_envvars_extends_the_cuda_list(platform):
    assert "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS" in platform.ray_noset_envvars()


def test_is_available_is_false_without_a_tpu_runtime(platform):
    assert platform.is_available() is False


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, False),
        ({"VERL_PLATFORM": "tpu"}, True),
        ({"VERL_PLATFORM": "nvidia"}, False),
        ({"TPU_NAME": "local"}, True),
        ({"TPU_VISIBLE_DEVICES": "0"}, True),
    ],
)
def test_is_platform_available_reads_the_environment(monkeypatch, platform, env, expected):
    for key in ("VERL_PLATFORM", "TPU_NAME", "TPU_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert platform.is_platform_available() is expected


def test_is_platform_available_falls_back_to_the_runtime(monkeypatch, platform):
    """With no TPU env vars set, a working runtime is enough to claim the platform.

    Nothing sets ``TPU_NAME`` on a plain TPU VM, so without this the auto-detection in
    ``_detect_platform_name`` finds no available platform and defaults to CUDA.
    """
    for key in ("VERL_PLATFORM", "TPU_NAME", "TPU_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(type(platform), "is_available", lambda self: True)

    assert platform.is_platform_available() is True


# ---------------------------------------------------------------------------
# Capabilities consumed by shared code
# ---------------------------------------------------------------------------


def test_tpu_chips_are_not_shareable_between_worker_groups(platform):
    assert platform.supports_colocated_worker_groups() is False


def test_other_platforms_still_allow_colocation():
    """The cap is opt-in; the default must not change for existing accelerators."""
    assert PlatformBase.supports_colocated_worker_groups(object()) is True


def test_local_rank_comes_from_tpu_visible_chips(monkeypatch, platform):
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "3")
    assert platform.ray_local_rank_override() == "3"


def test_local_rank_defaults_to_zero_when_unset(monkeypatch, platform):
    monkeypatch.delenv("TPU_VISIBLE_CHIPS", raising=False)
    assert platform.ray_local_rank_override() == "0"


def test_other_platforms_defer_to_ray_for_local_rank():
    """Returning None keeps Ray's accelerator-id path in charge for every other backend."""
    assert PlatformBase.ray_local_rank_override(object()) is None


def test_worker_env_vars_default_to_empty():
    assert PlatformBase.get_worker_env_vars(object(), None, 0, 1, 0, 1, "p", "cuda") == {}


# ---------------------------------------------------------------------------
# Device module proxy
# ---------------------------------------------------------------------------


def test_proxy_reports_zero_for_unsupported_memory_stats(platform):
    module = platform.device_module
    assert module.memory_reserved() == 0
    assert module.memory_allocated() == 0
    assert module.max_memory_reserved() == 0
    assert module.max_memory_allocated() == 0
    assert module.reset_peak_memory_stats() is None


def test_proxy_reports_hbm_capacity(platform):
    """verl reads total memory in generic code paths, so it has to be answerable."""
    total, free = platform.device_module.mem_get_info()
    assert total == free == HBM_BYTES_TPU_DEFAULT
    assert platform.device_module.get_device_properties().total_memory == HBM_BYTES_TPU_DEFAULT


def test_proxy_raises_for_genuinely_unknown_attributes(platform):
    with pytest.raises(AttributeError):
        _ = platform.device_module.no_such_attribute


def test_proxy_is_safe_over_the_dummy_module():
    """Driver and CPU-only processes get the dummy module; nothing may raise there."""
    proxy = TPUDeviceModuleProxy(DummyTpuDeviceModule())
    assert proxy.is_available() is False
    assert proxy.device_count() == 0
    assert proxy.current_device() == 0
    assert proxy.set_device(0) is None
    assert proxy.synchronize() is None
    assert proxy.empty_cache() is None


# ---------------------------------------------------------------------------
# Chip detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "accelerator_type,expected",
    [
        ("v6e-8", HBM_BYTES_TPU_V6E),
        ("V5P-16", HBM_BYTES_TPU_V5P),
        ("something-unknown", -1),
    ],
)
def test_hbm_capacity_detected_from_environment(monkeypatch, accelerator_type, expected):
    for key in ("ACCELERATOR_TYPE", "TPU_TYPE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", accelerator_type)

    assert get_tpu_chip_hbm_bytes() == expected
