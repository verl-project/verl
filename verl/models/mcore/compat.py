# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# there is some bug in mcore 0.12, so we need to patch it
# 1. `get_query_key_value_tensors` in `multi_latent_attention.py` works wrong when packed_seq_params is not None


"""Single installation boundary for temporary Megatron compatibility.

Only this module selects patch implementations. Retirement requires both native
capability and a consumer using that contract, not merely a merged upstream PR.
"""

import logging
from dataclasses import dataclass
from threading import RLock
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchInfo:
    """Reviewable purpose and removal condition for one compatibility feature."""

    reason: str
    upstream: tuple[str, ...]
    retire_when: str


PATCHES = {
    "flash_attn_cute": PatchInfo("Broken optional FA4 import", (), "Installed FA4 imports successfully"),
    "checkpoint_writer": PatchInfo(
        "MCore 0.12.1 / Torch 2.8-2.9 writer API", (), "Core or Torch is outside the affected versions"
    ),
    "hadamard": PatchInfo(
        "Optional CUDA extension unavailable on ROCm", (), "Native fast_hadamard_transform imports successfully"
    ),
    "mla": PatchInfo("Legacy MLA packed sequence execution", (), "MCore >= 0.16.2"),
    "router_replay": PatchInfo(
        "Legacy verl replay consumer",
        ("https://github.com/NVIDIA/Megatron-LM/pull/8674",),
        "Consumer migrates to native replay and replay_mask/exact recomputation are validated",
    ),
    "mtp_inference": PatchInfo(
        "Ascend zero-layer MTP compatibility", (), "Native zero-layer inference is validated; no verified boundary yet"
    ),
    "recompute": PatchInfo(
        "MoE checkpoint residual memory",
        ("https://github.com/verl-project/verl/pull/8338",),
        "Gradient and MoE memory parity pass; linked verl PR is not a Core fix",
    ),
    "mtp_postprocess": PatchInfo(
        "RL MTP loss, output weight and normalization",
        ("https://github.com/NVIDIA/Megatron-LM/pull/8673",),
        "Consumer uses native MTP controls and detach; #8673 covers detach only",
    ),
    "mtp_checkpoint": PatchInfo(
        "MTP checkpoint handling", (), "Native MTP checkpoint path passes supported runtime-CP cases"
    ),
    "mtp_embeddings": PatchInfo(
        "MTP backbone detach and runtime CP",
        (),
        "Native backbone detach/runtime CP passes; no complete upstream fix yet",
    ),
}

_checked_global = set()
_checked_models = WeakKeyDictionary()
_install_lock = RLock()


def install_patch(name, install, *, needed, owner=None):
    """Check necessity first; install at most once per process or model.

    ``needed`` is a callable so native capability is checked before touching the
    target. An installer may perform its own dependency probe, hence a true
    return means the compatibility check ran, not necessarily a method changed.
    Failed installs are not marked complete and can be retried.
    """
    info = PATCHES[name]
    with _install_lock:
        if not needed():
            logger.debug("Megatron compatibility %s skipped: %s", name, info.retire_when)
            return False
        installed = _checked_global if owner is None else _checked_models.setdefault(owner, set())
        if name in installed:
            return False
        install()
        installed.add(name)
        logger.info("Megatron compatibility %s checked: %s; upstream=%s", name, info.reason, info.upstream)
        return True


def _legacy(name):
    from verl.models.mcore import patch

    return getattr(patch, name)


def apply_bootstrap_patches():
    """Run before registry imports can reach Megatron optional extensions."""
    install_patch("flash_attn_cute", lambda: _legacy("neutralize_broken_flash_attn_cute")(), needed=lambda: True)

    def writer_affected():
        import megatron.core
        import torch
        from packaging.version import Version

        return Version(megatron.core.__version__).base_version == "0.12.1" and Version(
            torch.__version__
        ).base_version in ("2.8.0", "2.9.0")

    install_patch(
        "checkpoint_writer", lambda: _legacy("apply_patch_megatron_v012_with_torch_v28_v29")(), needed=writer_affected
    )


def apply_construction_patches(config):
    """Keep legacy MLA behavior while avoiding obsolete implementations on new Core."""
    import megatron.core
    from megatron.core.transformer import MLATransformerConfig
    from packaging.version import Version

    if not isinstance(config, MLATransformerConfig):
        return
    install_patch("hadamard", lambda: _legacy("apply_fast_hadamard_transform_shim")(), needed=lambda: True)
    install_patch(
        "mla", lambda: _legacy("apply_patch")(), needed=lambda: Version(megatron.core.__version__) < Version("0.16.2")
    )


def apply_engine_patches(*, routing_replay=False):
    """Install process-wide behavior requested by this engine."""
    import megatron.core
    from packaging.version import Version

    from verl.utils.device import is_cuda_available, is_npu_available

    def replay():
        from verl.utils.megatron.router_replay_patch import apply_router_replay_patch

        apply_router_replay_patch()

    # Native replay cannot replace the legacy driver's registry until slice 4.
    install_patch("router_replay", replay, needed=lambda: routing_replay)
    install_patch(
        "mtp_inference",
        lambda: _legacy("apply_mtp_inference_patch")(),
        needed=lambda: is_npu_available and Version(megatron.core.__version__) >= Version("0.16.0"),
    )
    # Do not guess a retirement version for the unresolved memory regression.
    install_patch(
        "recompute", lambda: _legacy("apply_patch_megatron_recomputation_backward")(), needed=lambda: is_cuda_available
    )


def apply_model_patches(models, *, mtp_enabled=False, detach_encoder=False, forward_only=False, mtp_num_layers=None):
    """Install only per-model MTP compatibility, without stacking replacements."""
    from verl.models.mcore import mtp_patch

    for model in models if isinstance(models, list | tuple) else [models]:
        postprocess = mtp_enabled or (forward_only and mtp_num_layers == 0)
        install_patch(
            "mtp_postprocess",
            lambda model=model: mtp_patch.patch_postprocess(model),
            needed=lambda postprocess=postprocess: postprocess,
            owner=model,
        )
        install_patch(
            "mtp_checkpoint",
            lambda model=model: mtp_patch.patch_mtp_layer_checkpointed_forward(model),
            needed=lambda: mtp_enabled,
            owner=model,
        )
        install_patch(
            "mtp_embeddings",
            lambda model=model: mtp_patch.patch_mtp_layer_get_embeddings(model),
            needed=lambda: mtp_enabled and detach_encoder,
            owner=model,
        )


def configure_forward(model, model_config=None, *, engine_config=None):
    """Select native output hooks or the isolated legacy forward compatibility."""
    from verl.models.mcore.model_forward_fused import patch_fused_forward

    return patch_fused_forward(model, model_config, engine_config=engine_config)


def has_output_processor(model):
    """Check the complete callback contract, not a version string or **kwargs."""
    import inspect

    from verl.utils.megatron.native import get_language_model

    model = get_language_model(model)
    # The retained MTP postprocess accepts these kwargs but does not execute the
    # callback. It must stay on eager output until the MTP migration (slice 3).
    postprocess = getattr(model, "_postprocess", None)
    if getattr(postprocess, "__module__", None) == "verl.models.mcore.mtp_patch":
        return False
    parameters = inspect.signature(model.forward).parameters
    return {"output_processor", "output_processor_context"}.issubset(parameters)


def _readiness_api():
    from megatron.core import utils

    native = getattr(utils, "ensure_params_ready", None)
    if native is not None:
        return lambda weight: native([weight])
    try:
        from verl.utils.megatron.param_sync import ensure_fused_weight_ready
    except ImportError:
        return None
    return ensure_fused_weight_ready


def supports_fused_readiness(model):
    """Old Core overlap needs existing verl PR #8029; native Core bypasses it."""
    if _readiness_api() is not None:
        return True
    while True:
        if getattr(getattr(model, "ddp_config", None), "overlap_param_gather", False):
            return False
        if not hasattr(model, "module"):
            return True
        model = model.module


def ensure_output_weight_ready(weight):
    """Use native readiness, or the helper from #8029 once it is available."""
    ready = _readiness_api()
    if ready is not None:
        ready(weight)
