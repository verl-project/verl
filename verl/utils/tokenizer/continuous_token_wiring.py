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
"""Continuous Token builder factory and model-family resolution."""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

from .continuous_token import (
    ContinuousTokenBuilder,
    DeepSeekContinuousTokenBuilder,
    DeepSeekVL2ContinuousTokenBuilder,
    Gemma4ContinuousTokenBuilder,
    Gemma4VLContinuousTokenBuilder,
    GLM46VContinuousTokenBuilder,
    GLMContinuousTokenBuilder,
    GptOssContinuousTokenBuilder,
    KimiVLContinuousTokenBuilder,
    MiniMaxContinuousTokenBuilder,
    MiniMaxVLContinuousTokenBuilder,
    Qwen3ContinuousTokenBuilder,
    Qwen35ContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
    QwenVLContinuousTokenBuilder,
    VLContinuousTokenBuilder,
)
from .deepseek import DeepSeekV4ContinuousTokenBuilder

logger = logging.getLogger(__name__)


class ContinuousTokenModelFamily(str, Enum):
    # ``enum.StrEnum`` needs Python 3.11, but verl supports 3.10, so mix in ``str`` and keep its
    # ``__str__``/``__format__`` to render a member as its value.
    __str__ = str.__str__
    __format__ = str.__format__

    AUTO = "auto"
    DEFAULT = "default"
    QWEN = "qwen"
    QWEN25 = "qwen25"
    QWEN3 = "qwen3"
    QWEN35 = "qwen35"
    MINIMAX = "minimax"
    MINIMAX_M2 = "minimaxm2"
    MINIMAX_M25 = "minimaxm25"
    MINIMAX_M27 = "minimaxm27"
    GLM47 = "glm47"
    GLM5 = "glm5"
    GEMMA4 = "gemma4"
    GPTOSS = "gptoss"
    DEEPSEEK = "deepseek"
    # Multimodal (VL) families
    VL_DEFAULT = "vldefault"
    QWEN_VL = "qwenvl"
    QWEN25_VL = "qwen25vl"
    QWEN3_VL = "qwen3vl"
    MINIMAX_VL = "minimaxvl"
    GEMMA4_VL = "gemma4vl"
    KIMI_VL = "kimivl"
    GLM4V = "glm4v"
    DEEPSEEK_VL2 = "deepseekvl2"
    DEEPSEEKV4 = "deepseekv4"


_CONTINUOUS_TOKEN_BUILDER_REGISTRY: dict[ContinuousTokenModelFamily, type[Any]] = {
    ContinuousTokenModelFamily.DEFAULT: ContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN: QwenContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN25: QwenContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN3: Qwen3ContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN35: Qwen35ContinuousTokenBuilder,
    ContinuousTokenModelFamily.MINIMAX: MiniMaxContinuousTokenBuilder,
    ContinuousTokenModelFamily.MINIMAX_M2: MiniMaxContinuousTokenBuilder,
    ContinuousTokenModelFamily.MINIMAX_M25: MiniMaxContinuousTokenBuilder,
    ContinuousTokenModelFamily.MINIMAX_M27: MiniMaxContinuousTokenBuilder,
    ContinuousTokenModelFamily.GLM47: GLMContinuousTokenBuilder,
    ContinuousTokenModelFamily.GLM5: GLMContinuousTokenBuilder,
    ContinuousTokenModelFamily.GEMMA4: Gemma4ContinuousTokenBuilder,
    ContinuousTokenModelFamily.GPTOSS: GptOssContinuousTokenBuilder,
    ContinuousTokenModelFamily.DEEPSEEK: DeepSeekContinuousTokenBuilder,
    # Multimodal (VL) families
    ContinuousTokenModelFamily.VL_DEFAULT: VLContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN_VL: QwenVLContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN25_VL: QwenVLContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN3_VL: QwenVLContinuousTokenBuilder,
    ContinuousTokenModelFamily.MINIMAX_VL: MiniMaxVLContinuousTokenBuilder,
    ContinuousTokenModelFamily.GEMMA4_VL: Gemma4VLContinuousTokenBuilder,
    ContinuousTokenModelFamily.KIMI_VL: KimiVLContinuousTokenBuilder,
    ContinuousTokenModelFamily.GLM4V: GLM46VContinuousTokenBuilder,
    ContinuousTokenModelFamily.DEEPSEEK_VL2: DeepSeekVL2ContinuousTokenBuilder,
    ContinuousTokenModelFamily.DEEPSEEKV4: DeepSeekV4ContinuousTokenBuilder,
}

CONTINUOUS_TOKEN_BUILDER_FAMILIES = tuple(family.value for family in _CONTINUOUS_TOKEN_BUILDER_REGISTRY)

# Exact Hugging Face ``config.json`` model_type -> Continuous Token protocol.
# Only model types validated against a model-specific builder belong here. Unknown
# values deliberately use the generic text/VL builder rather than being guessed
# from repository names or implementation class names.
_MODEL_TYPE_TO_FAMILY: dict[str, ContinuousTokenModelFamily] = {
    # Text models.
    "qwen2": ContinuousTokenModelFamily.QWEN,
    "qwen3": ContinuousTokenModelFamily.QWEN3,
    "qwen3_moe": ContinuousTokenModelFamily.QWEN3,
    "qwen3_5": ContinuousTokenModelFamily.QWEN35,
    "qwen3_5_moe": ContinuousTokenModelFamily.QWEN35,
    "minimax": ContinuousTokenModelFamily.MINIMAX,
    "minimax_text_01": ContinuousTokenModelFamily.MINIMAX,
    # MiniMax M2-series checkpoints share this root model_type. Exact config
    # matching therefore selects their shared builder without guessing a minor
    # version from the repository path.
    "minimax_m2": ContinuousTokenModelFamily.MINIMAX_M2,
    "glm4_moe": ContinuousTokenModelFamily.GLM47,
    "glm_moe_dsa": ContinuousTokenModelFamily.GLM5,
    "gemma4": ContinuousTokenModelFamily.GEMMA4,
    "gpt_oss": ContinuousTokenModelFamily.GPTOSS,
    "deepseek_v2": ContinuousTokenModelFamily.DEEPSEEK,
    "deepseek_v3": ContinuousTokenModelFamily.DEEPSEEK,
    "deepseek_v4": ContinuousTokenModelFamily.DEEPSEEKV4,
    # Vision-language models. The processor is still required at construction.
    "qwen2_vl": ContinuousTokenModelFamily.QWEN_VL,
    "qwen2_5_vl": ContinuousTokenModelFamily.QWEN25_VL,
    "qwen3_vl": ContinuousTokenModelFamily.QWEN3_VL,
    "qwen3_vl_moe": ContinuousTokenModelFamily.QWEN3_VL,
    "minimax_vl_01": ContinuousTokenModelFamily.MINIMAX_VL,
    "kimi_vl": ContinuousTokenModelFamily.KIMI_VL,
    "glm4v": ContinuousTokenModelFamily.GLM4V,
    "glm4v_moe": ContinuousTokenModelFamily.GLM4V,
    "deepseek_vl_v2": ContinuousTokenModelFamily.DEEPSEEK_VL2,
}

# Unified checkpoints whose root model type is shared by their text and vision
# modes. A multimodal processor selects the processor-backed builder.
_TEXT_TO_VL_FAMILY: dict[ContinuousTokenModelFamily, ContinuousTokenModelFamily] = {
    ContinuousTokenModelFamily.DEFAULT: ContinuousTokenModelFamily.VL_DEFAULT,
    ContinuousTokenModelFamily.GEMMA4: ContinuousTokenModelFamily.GEMMA4_VL,
    ContinuousTokenModelFamily.QWEN35: ContinuousTokenModelFamily.QWEN3_VL,
}


def get_continuous_token_builder_class(model_family: str | ContinuousTokenModelFamily) -> type[Any]:
    family = _normalize_model_family(model_family)
    try:
        return _CONTINUOUS_TOKEN_BUILDER_REGISTRY[family]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Continuous Token builder family {family!r}. "
            f"Supported families: {CONTINUOUS_TOKEN_BUILDER_FAMILIES}."
        ) from exc


def list_continuous_token_builder_families() -> tuple[str, ...]:
    return CONTINUOUS_TOKEN_BUILDER_FAMILIES


def resolve_continuous_token_model_family(
    model_family: str | ContinuousTokenModelFamily,
    *,
    hf_model_type: str | None = None,
    has_multimodal_processor: bool = False,
) -> ContinuousTokenModelFamily:
    """Resolve ``auto`` to a concrete family, or canonicalize an explicit family."""
    family = _normalize_model_family(model_family)
    if family != ContinuousTokenModelFamily.AUTO:
        logger.info("Using explicit Continuous Token builder family: %s", family)
        return family

    resolved = infer_continuous_token_model_family(
        hf_model_type=hf_model_type,
        has_multimodal_processor=has_multimodal_processor,
    )
    logger.info(
        "Resolved Continuous Token builder family from config.json model_type=%r: %s",
        _normalize_hf_model_type(hf_model_type),
        resolved,
    )
    return resolved


def infer_continuous_token_model_family(
    *,
    hf_model_type: str | None = None,
    has_multimodal_processor: bool = False,
) -> ContinuousTokenModelFamily:
    """Infer a builder from the root ``config.json`` ``model_type`` field.

    Matching is exact. Repository paths, tokenizer names, nested configs, and
    ``architectures`` are intentionally ignored. Unknown model types use the
    generic builder selected by whether a multimodal processor is present.
    """
    model_type = _normalize_hf_model_type(hf_model_type)
    if model_type is not None:
        family = _MODEL_TYPE_TO_FAMILY.get(model_type)
        if family is not None:
            return family

    fallback = ContinuousTokenModelFamily.VL_DEFAULT if has_multimodal_processor else ContinuousTokenModelFamily.DEFAULT
    logger.warning(
        "No model-specific Continuous Token builder is registered for config.json model_type=%r; falling back to %s.",
        model_type,
        fallback,
    )
    return fallback


def create_continuous_token_builder(
    tokenizer: Any,
    *,
    model_family: str | ContinuousTokenModelFamily = "auto",
    hf_model_type: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
    processor: Any | None = None,
    **builder_kwargs: Any,
) -> Any:
    """Instantiate the Continuous Token builder inferred from the root Hugging Face ``model_type``.

    Inference uses an exact registry lookup on the root Hugging Face config's
    ``model_type``. Repository/tokenizer names and ``architectures`` are not used.
    Whether an unknown model gets the generic text or VL builder is decided by the
    presence of a multimodal ``processor``.

    Resolution rules:
      * Text (no multimodal processor): use the inferred model-specific text
        builder, or the default builder when nothing matched (a warning is emitted
        by the inference step in that case).
      * VL (multimodal processor present):
          - If ``model_type`` resolves to a VL family, use that VL builder.
          - If it resolves to a unified text family, upgrade to its VL builder.
          - If it is unknown, use the default VL builder and warn.
          - Any other recognized text-specific family paired with a processor is
            treated as a misconfiguration and raises.
    """
    has_mm_processor = _is_multimodal_processor(processor)
    resolved_family = resolve_continuous_token_model_family(
        model_family,
        hf_model_type=hf_model_type,
        has_multimodal_processor=has_mm_processor,
    )
    builder_cls = get_continuous_token_builder_class(resolved_family)

    if has_mm_processor:
        # --- Vision-language run ---
        # mm_processor_kwargs is a multimodal-only concern, so (like ``processor``) it
        # is passed only to VL builders; text builders never receive it.
        if builder_cls.supports_multimodal():
            # The root model_type identified a model-specific VL family.
            logger.info("Creating Continuous Token builder: family=%s class=%s", resolved_family, builder_cls)
            return builder_cls(
                tokenizer,
                processor,
                chat_template_kwargs=chat_template_kwargs,
                mm_processor_kwargs=mm_processor_kwargs,
                **builder_kwargs,
            )

        # Inferred a text family, but a multimodal processor is present. Only
        # explicitly unified families are safe to auto-upgrade.
        if resolved_family in _TEXT_TO_VL_FAMILY:
            upgraded_family = _TEXT_TO_VL_FAMILY[resolved_family]
            if upgraded_family == ContinuousTokenModelFamily.VL_DEFAULT:
                logger.warning(
                    "No model-specific VL Continuous Token builder matched (inferred %s); "
                    "falling back to the default VL builder (VLContinuousTokenBuilder).",
                    resolved_family,
                )
            else:
                logger.info(
                    "Multimodal processor detected with unified family %s; upgrading to VL family %s.",
                    resolved_family,
                    upgraded_family,
                )
            resolved_family = upgraded_family
            builder_cls = get_continuous_token_builder_class(resolved_family)
            logger.info("Creating Continuous Token builder: family=%s class=%s", resolved_family, builder_cls)
            return builder_cls(
                tokenizer,
                processor,
                chat_template_kwargs=chat_template_kwargs,
                mm_processor_kwargs=mm_processor_kwargs,
                **builder_kwargs,
            )

        raise ValueError(
            f"Model resolved to the text Continuous Token family {resolved_family!r}, but a multimodal "
            f"processor was provided. Register config.json model_type={_normalize_hf_model_type(hf_model_type)!r} "
            f"as a VL or unified family, or do not load a multimodal processor."
        )

    # --- Text-only run (no multimodal processor) ---
    if builder_cls.supports_multimodal():
        raise ValueError(
            f"Model resolved to the VL Continuous Token family {resolved_family!r} "
            f"({builder_cls.__name__}), which requires a processor, but none was provided. "
            f"Ensure the processor is loaded for vision-language models."
        )
    logger.info("Creating Continuous Token builder: family=%s class=%s", resolved_family, builder_cls)
    return builder_cls(tokenizer, chat_template_kwargs=chat_template_kwargs, **builder_kwargs)


def _is_multimodal_processor(processor: Any | None) -> bool:
    """Whether ``processor`` is a multimodal processor (has an image processor)."""
    return processor is not None and getattr(processor, "image_processor", None) is not None


def _normalize_hf_model_type(hf_model_type: str | None) -> str | None:
    """Normalize a root Hugging Face ``model_type`` value."""
    if not isinstance(hf_model_type, str):
        return None
    normalized = hf_model_type.strip().lower()
    return normalized or None


def _normalize_model_family(model_family: str | ContinuousTokenModelFamily) -> ContinuousTokenModelFamily:
    if isinstance(model_family, ContinuousTokenModelFamily):
        return model_family
    if not isinstance(model_family, str) or not model_family:
        raise ValueError("Continuous Token model_family must be a non-empty string")
    family = model_family.strip().lower()
    if not family:
        raise ValueError("Continuous Token model_family must be a non-empty string")
    family = re.sub(r"[^a-z0-9]+", "", family)
    try:
        return ContinuousTokenModelFamily(family)
    except ValueError as exc:
        raise ValueError(
            f"Unknown Continuous Token model_family {model_family!r}. "
            f"Supported families: {(ContinuousTokenModelFamily.AUTO.value, *CONTINUOUS_TOKEN_BUILDER_FAMILIES)}."
        ) from exc
