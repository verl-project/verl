"""MaxText configuration and JAX environment for the engine.

The trainer configuration is a plain MaxText config: a YAML file that MaxText
itself would accept (``python -m maxtext.trainers.pre_train.train my.yml``),
plus ``key=value`` overrides. Nothing here depends on another trainer launcher.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

import yaml

# Applied only when the variable is not already set, so a launcher or the image wins.
_ENV_DEFAULTS = {
    "XLA_PYTHON_CLIENT_MEM_FRACTION": ".97",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
    "NVTE_USE_HIPBLASLT": "1",
    "NVTE_FUSED_ATTN": "1",
    "NVTE_FUSED_ATTN_CK": "1",
    "NVTE_FUSED_ATTN_AOTRITON": "0",
    "NVTE_CK_USES_BWD_V3": "1",
    "NVTE_CK_USES_FWD_V3": "1",
    "NVTE_CK_IS_V3_ATOMIC_FP32": "0",
    "NVTE_CK_HOW_V3_BF16_CVT": "2",
    "GPU_MAX_HW_QUEUES": "2",
    # Newer HIP ignores stream priority by default, which removes comm/compute overlap.
    "DEBUG_HIP_IGNORE_STREAM_PRIORITY": "0",
    "HIP_FORCE_DEV_KERNARG": "1",
    "HSA_FORCE_FINE_GRAIN_PCIE": "1",
}

# Managed XLA flags. A flag given here replaces the same flag already in XLA_FLAGS.
_XLA_FLAGS = (
    "--xla_gpu_memory_limit_slop_factor=95",
    "--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592",
    "--xla_gpu_enable_command_buffer=''",
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_all_gather_combine_threshold_bytes=8589934592",
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_enable_cublaslt=true",
    # Autotuning must stay on: with it off, fp8 MoE expert GEMMs can overflow.
    "--xla_gpu_autotune_level={autotune_level}",
    "--xla_gpu_enable_all_gather_combine_by_dim=false",
    "--xla_gpu_enable_dynamic_slice_fusion=false",
    # The hipCUB segmented sort behind MoE top-k does a pageable host copy per call.
    "--xla_gpu_enable_cub_radix_sort=false",
)


def _gpu_arch() -> str | None:
    tool = shutil.which("rocm_agent_enumerator")
    if tool is None:
        return None
    try:
        out = subprocess.run([tool], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    arches = [line.strip() for line in out.splitlines() if line.strip().startswith("gfx") and line.strip() != "gfx000"]
    return arches[0] if arches else None


def _merge_xla_flags(existing: str, managed: list[str]) -> str:
    names = {flag.split("=", 1)[0] for flag in managed}
    kept = [tok for tok in re.findall(r"\S+", existing) if tok.split("=", 1)[0] not in names]
    return " ".join(kept + managed)


def prepare_jax_env(config) -> None:
    """Set ROCm/XLA tuning variables before JAX is imported.

    ``config.env`` entries always win. ``mem_fraction`` is applied last so some
    device memory stays free for the torch tensors the rollout sync creates.
    """
    arch = _gpu_arch()
    defaults = dict(_ENV_DEFAULTS)
    if arch == "gfx950":
        defaults["RCCL_WARP_SPEED_AUTO"] = "0"
    elif arch in (None, "gfx942"):
        defaults["HSA_NO_SCRATCH_RECLAIM"] = "1"
    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    autotune = os.environ.get("XLA_GPU_AUTOTUNE_LEVEL", "4")
    managed = [flag.format(autotune_level=autotune) for flag in _XLA_FLAGS]
    os.environ["XLA_FLAGS"] = _merge_xla_flags(os.environ.get("XLA_FLAGS", ""), managed)

    for key, value in dict(getattr(config, "env", {}) or {}).items():
        os.environ[str(key)] = str(value)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.mem_fraction)


def bind_maxtext(config) -> None:
    """Make ``import maxtext`` resolve to the checkout in ``maxtext_path`` or ``$MAXTEXT_PATH``."""
    root = config.maxtext_path or os.environ.get("MAXTEXT_PATH")
    if root:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"MaxText checkout not found: {root}")
        for candidate in (os.path.join(root, "src"), root):
            if os.path.isdir(os.path.join(candidate, "maxtext")) and candidate not in sys.path:
                sys.path.insert(0, candidate)
    importlib.invalidate_caches()
    if importlib.util.find_spec("maxtext") is None:
        raise ModuleNotFoundError("Install MaxText or set maxtext_path / MAXTEXT_PATH to a MaxText checkout")


def _default_base_config() -> str:
    spec = importlib.util.find_spec("maxtext")
    package = os.path.dirname(spec.origin) if spec and spec.origin else None
    path = os.path.join(package or "", "configs", "base.yml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MaxText base.yml not found at {path}; set config_path")
    return path


def _format_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list | tuple | dict):
        return json.dumps(value)
    return str(value)


def _user_values(config_path: str, overrides: dict) -> dict:
    """Keys set by the config file chain and the overrides, for defaults that depend on other keys.

    ``base_config`` is followed the way MaxText does: relative to the including
    file first. A parent that is not next to it is MaxText's own and is skipped.
    """
    with open(config_path) as handle:
        values = yaml.safe_load(handle) or {}
    parent = values.get("base_config")
    if parent and not os.path.isabs(parent):
        parent = os.path.join(os.path.dirname(config_path), parent)
    if parent and os.path.isfile(parent):
        values = {**_user_values(parent, {}), **values}
    return {**values, **overrides}


def maxtext_argv(config) -> list[str]:
    """``argv`` for ``maxtext.trainers.pre_train.train.initialize``.

    The first entry after the program name is the config file; the rest are
    ``key=value`` overrides, which MaxText applies on top of it.
    """
    bind_maxtext(config)
    overrides = dict(getattr(config, "overrides", {}) or {})
    path = config.config_path
    if path:
        path = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"MaxText config not found: {path}")
        if not path.endswith(".yml"):
            # MaxText only treats an argv entry ending in .yml as the config file.
            raise ValueError(f"MaxText config files must end in .yml, got {path}")
    else:
        path = _default_base_config()
    values = _user_values(path, overrides)
    # MaxText writes device_info.json here at startup and does not create the directory.
    output_dir = values.get("base_output_directory")
    if output_dir and "://" not in str(output_dir):
        os.makedirs(str(output_dir), exist_ok=True)
    # An unset or zero eval batch makes MaxText size the eval pipeline to nothing.
    per_device = values.get("per_device_batch_size")
    if per_device is not None and float(values.get("eval_per_device_batch_size") or 0) <= 0:
        overrides["eval_per_device_batch_size"] = per_device
    return ["maxtext", path] + [f"{key}={_format_override(value)}" for key, value in overrides.items()]
