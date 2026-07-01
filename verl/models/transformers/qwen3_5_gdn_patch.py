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

import os
from collections.abc import Callable
from typing import Any

_GDN_BACKEND_ENV = "VERL_NPU_QWEN3_5_GDN_BACKEND"
_TRITON_ALIASES = {"1", "true", "on", "enable", "enabled", "auto", "triton"}
_EAGER_ALIASES = {"0", "false", "off", "disable", "disabled", "eager", "torch", "none"}


def _canonicalize_gdn_backend(backend: str) -> str:
    if backend in _TRITON_ALIASES:
        return "triton"
    if backend in _EAGER_ALIASES:
        return "eager"
    return backend


def _normalize_gdn_backend(config: Any) -> str:
    env_backend = os.getenv(_GDN_BACKEND_ENV, "").strip().lower()
    if env_backend:
        return _canonicalize_gdn_backend(env_backend)

    backend = getattr(config, "gdn_compute_mode", None)
    if backend is not None:
        return _canonicalize_gdn_backend(str(backend).strip().lower())

    if getattr(config, "use_triton_gdn", False):
        return "triton"
    return "eager"


def _restore_use_triton_gdn(config: Any, had_attr: bool, original_value: Any) -> None:
    if had_attr:
        setattr(config, "use_triton_gdn", original_value)
    elif hasattr(config, "use_triton_gdn"):
        delattr(config, "use_triton_gdn")


def _load_triton_gdn() -> Callable[..., Any]:
    from verl.models.transformers.ops.gdn.chunk_gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def patch_qwen3_5_gdn_class(gdn_cls: type, *, logger: Any = None, class_name: str | None = None) -> None:
    original_init = gdn_cls.__init__
    if getattr(original_init, "_verl_npu_gdn_patched", False):
        return

    patched_name = class_name or gdn_cls.__name__

    def patched_init(self, config, *args, **kwargs):
        backend = _normalize_gdn_backend(config)
        if backend == "ascendc":
            raise ValueError(
                "gdn_compute_mode='ascendc' is not wired in verl yet. "
                "Use gdn_compute_mode='triton' or use_triton_gdn=True for the NPU Triton fused GDN path."
            )
        if backend not in {"triton", "eager"}:
            raise ValueError(
                f"Unsupported Qwen3.5 GDN backend '{backend}'. "
                "Expected one of: eager, triton, ascendc."
            )

        had_use_triton_gdn = hasattr(config, "use_triton_gdn")
        original_use_triton_gdn = getattr(config, "use_triton_gdn", None)
        setattr(config, "use_triton_gdn", False)

        try:
            original_init(self, config, *args, **kwargs)
        except Exception:
            _restore_use_triton_gdn(config, had_use_triton_gdn, original_use_triton_gdn)
            raise

        if backend == "triton":
            self.chunk_gated_delta_rule = _load_triton_gdn()
            self.use_triton_gdn = True
            setattr(config, "use_triton_gdn", True)
            if logger is not None:
                logger.info("%s uses verl NPU Triton GDN fused kernels", patched_name)
        else:
            setattr(config, "use_triton_gdn", False)

    patched_init._verl_npu_gdn_patched = True
    patched_init._verl_npu_gdn_original_init = original_init
    gdn_cls.__init__ = patched_init
