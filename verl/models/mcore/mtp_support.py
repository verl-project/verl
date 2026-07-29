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
"""Compatibility helpers for Megatron-Core's native HybridModel MTP path."""

from functools import lru_cache
from inspect import signature


@lru_cache(maxsize=1)
def has_native_mtp_support() -> bool:
    """Return whether MCore can derive MTP labels and detach MTP heads natively."""
    try:
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.multi_token_prediction import process_mtp_loss

        config_fields = getattr(TransformerConfig, "__dataclass_fields__", {})
        has_detach_heads = hasattr(TransformerConfig, "mtp_detach_heads") or "mtp_detach_heads" in config_fields
        return has_detach_heads and "input_ids" in signature(process_mtp_loss).parameters
    except (ImportError, TypeError, ValueError):
        return False


def configure_native_hybrid_mtp(provider, mtp_config, transformer_overrides: dict) -> bool:
    """Validate and configure native HybridModel MTP before model construction."""
    is_hybrid_provider = bool(
        getattr(provider, "is_hybrid_model", False) or type(provider).__name__ == "HybridModelProvider"
    )
    if not mtp_config.enable or not is_hybrid_provider:
        return False

    if not mtp_config.enable_train:
        raise ValueError(
            "HybridModel does not support model.mtp.enable=True with model.mtp.enable_train=False "
            "in this Megatron-Core version. Disable MTP entirely or enable MTP training."
        )
    if not has_native_mtp_support():
        raise RuntimeError(
            "HybridModel MTP training requires a Megatron-Core version with native "
            "process_mtp_loss(input_ids=...) and TransformerConfig.mtp_detach_heads support."
        )

    transformer_overrides["mtp_detach_heads"] = bool(mtp_config.detach_encoder)
    return True


def is_native_hybrid_model(model) -> bool:
    """Return whether ``model`` is a HybridModel with the required native MTP API."""
    if not has_native_mtp_support():
        return False
    try:
        from megatron.core.models.hybrid.hybrid_model import HybridModel
    except ImportError:
        return False
    return isinstance(model, HybridModel)
