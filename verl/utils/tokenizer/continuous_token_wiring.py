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
from enum import StrEnum
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
    MiMoContinuousTokenBuilder,
    MiMoVLContinuousTokenBuilder,
    MiniMaxContinuousTokenBuilder,
    MiniMaxVLContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
    QwenVLContinuousTokenBuilder,
    VLContinuousTokenBuilder,
)

logger = logging.getLogger(__name__)


class ContinuousTokenModelFamily(StrEnum):
    AUTO = "auto"
    DEFAULT = "default"
    QWEN = "qwen"
    QWEN25 = "qwen25"
    QWEN3 = "qwen3"
    QWEN35 = "qwen35"
    MIMO = "mimo"
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
    MIMO_VL = "mimovl"
    MINIMAX_VL = "minimaxvl"
    GEMMA4_VL = "gemma4vl"
    KIMI_VL = "kimivl"
    GLM4V = "glm4v"
    DEEPSEEK_VL2 = "deepseekvl2"


_CONTINUOUS_TOKEN_BUILDER_REGISTRY: dict[ContinuousTokenModelFamily, type[Any]] = {
    ContinuousTokenModelFamily.DEFAULT: ContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN: QwenContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN25: QwenContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN3: QwenContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN35: QwenContinuousTokenBuilder,
    ContinuousTokenModelFamily.MIMO: MiMoContinuousTokenBuilder,
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
    ContinuousTokenModelFamily.MIMO_VL: MiMoVLContinuousTokenBuilder,
    ContinuousTokenModelFamily.MINIMAX_VL: MiniMaxVLContinuousTokenBuilder,
    ContinuousTokenModelFamily.GEMMA4_VL: Gemma4VLContinuousTokenBuilder,
    ContinuousTokenModelFamily.KIMI_VL: KimiVLContinuousTokenBuilder,
    ContinuousTokenModelFamily.GLM4V: GLM46VContinuousTokenBuilder,
    ContinuousTokenModelFamily.DEEPSEEK_VL2: DeepSeekVL2ContinuousTokenBuilder,
}

CONTINUOUS_TOKEN_BUILDER_FAMILIES = tuple(family.value for family in _CONTINUOUS_TOKEN_BUILDER_REGISTRY)

# Unified checkpoints whose model name carries no ``vl`` marker (so name-based
# inference resolves to the text family) but that can still be driven in vision
# mode. When a multimodal processor is supplied we upgrade the text family to its
# VL counterpart so rendering goes through the processor chat template.
_TEXT_TO_VL_FAMILY: dict[ContinuousTokenModelFamily, ContinuousTokenModelFamily] = {
    ContinuousTokenModelFamily.DEFAULT: ContinuousTokenModelFamily.VL_DEFAULT,
    ContinuousTokenModelFamily.GEMMA4: ContinuousTokenModelFamily.GEMMA4_VL,
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
    model_path: str | None = None,
    tokenizer: Any | None = None,
    tokenizer_name_or_path: str | None = None,
) -> ContinuousTokenModelFamily:
    """Resolve ``auto`` to a concrete family, or canonicalize an explicit family."""
    family = _normalize_model_family(model_family)
    if family != ContinuousTokenModelFamily.AUTO:
        logger.info("Using explicit Continuous Token builder family: %s", family)
        return family

    resolved = infer_continuous_token_model_family(
        model_path=model_path,
        tokenizer=tokenizer,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    logger.info(
        "Resolved Continuous Token builder family from auto: %s (model_path=%r, tokenizer_name_or_path=%r)",
        resolved,
        model_path,
        tokenizer_name_or_path or _tokenizer_name_or_path(tokenizer),
    )
    return resolved


def infer_continuous_token_model_family(
    *,
    model_path: str | None = None,
    tokenizer: Any | None = None,
    tokenizer_name_or_path: str | None = None,
) -> ContinuousTokenModelFamily:
    """Infer a built-in model family from model/tokenizer names.

    Unknown models intentionally fall back to ``default`` so enabling
    ``model_family=auto`` remains conservative.
    """
    candidates = [model_path, tokenizer_name_or_path, _tokenizer_name_or_path(tokenizer)]
    haystack = " ".join(str(item).lower() for item in candidates if item)
    compact = re.sub(r"[^a-z0-9]+", "", haystack)

    # --- VL families (must match before text families) ---
    # MiMo-VL
    if any(marker in haystack for marker in ("mimo-vl", "mimo_vl", "mimovl")):
        return ContinuousTokenModelFamily.MIMO_VL
    # MiniMax-VL (e.g. MiniMax-VL-01) — match before text minimax families
    if any(marker in haystack for marker in ("minimax-vl", "minimax_vl")) or "minimaxvl" in compact:
        return ContinuousTokenModelFamily.MINIMAX_VL
    # Qwen3-VL / Qwen3-VL-MoE
    if any(marker in haystack for marker in ("qwen3-vl", "qwen3_vl")) or "qwen3vl" in compact:
        return ContinuousTokenModelFamily.QWEN3_VL
    # Qwen2.5-VL
    if any(marker in haystack for marker in ("qwen2.5-vl", "qwen2_5-vl", "qwen2_5_vl")) or "qwen25vl" in compact:
        return ContinuousTokenModelFamily.QWEN25_VL
    # Qwen2-VL (also routes to QWEN_VL)
    if any(marker in haystack for marker in ("qwen2-vl", "qwen2_vl")) or "qwen2vl" in compact:
        return ContinuousTokenModelFamily.QWEN_VL
    # Kimi-VL
    if any(marker in haystack for marker in ("kimi-vl", "kimi_vl")) or "kimivl" in compact:
        return ContinuousTokenModelFamily.KIMI_VL
    # GLM vision editions (GLM-4V / GLM-4.1V / GLM-4.5V / GLM-4.6V) all share the
    # GLM-4.6V Continuous Token builder. GLM-4.6V is fully supported (single-turn
    # + tool agent loop). GLM-4V (4.1V) and GLM-4.5V are supported for the single
    # turn agent loop only: their templates mishandle tool-role images, so they
    # cannot be used with the tool agent loop, but single-turn prompt-image
    # rendering works fine through the same builder.
    if any(
        marker in haystack
        for marker in (
            "glm-4v",
            "glm4v",
            "glm-4.1v",
            "glm-4.1-vl",
            "glm-4.5v",
            "glm-4.5-vl",
            "glm-4.6v",
            "glm-4.6-vl",
        )
    ):
        return ContinuousTokenModelFamily.GLM4V

    # --- Existing families ---
    if any(marker in haystack for marker in ("glm-5", "glm_5")) or "glm5" in compact:
        return ContinuousTokenModelFamily.GLM5
    if any(marker in haystack for marker in ("glm-4.7", "glm_4.7", "glm4.7")) or "glm47" in compact:
        return ContinuousTokenModelFamily.GLM47
    if any(marker in haystack for marker in ("gemma-4", "gemma_4")) or any(
        marker in compact for marker in ("gemma4", "gemma4unified")
    ):
        return ContinuousTokenModelFamily.GEMMA4
    if any(marker in haystack for marker in ("gpt-oss", "gpt_oss")) or "gptoss" in compact:
        return ContinuousTokenModelFamily.GPTOSS
    # DeepSeek-VL2
    if "deepseek" in compact and "vl" in compact:
        return ContinuousTokenModelFamily.DEEPSEEK_VL2
    # DeepSeek text models (V2/V3/R1) — match if not VL2
    if "deepseek" in compact and "vl" not in compact:
        return ContinuousTokenModelFamily.DEEPSEEK
    # MiMo text (e.g. MiMo-7B-RL/SFT); MiMo-VL is already handled above. MiMo uses
    # a Qwen-style ChatML template, so it needs the same <|im_end|> newline patch.
    if "mimo" in compact:
        return ContinuousTokenModelFamily.MIMO
    if "minimaxm27" in compact:
        return ContinuousTokenModelFamily.MINIMAX_M27
    if "minimaxm25" in compact:
        return ContinuousTokenModelFamily.MINIMAX_M25
    if "minimaxm2" in compact:
        return ContinuousTokenModelFamily.MINIMAX_M2
    if "minimax" in compact:
        return ContinuousTokenModelFamily.MINIMAX
    if any(marker in haystack for marker in ("qwen3.5", "qwen3_5", "qwen3-5")) or "qwen35" in compact:
        return ContinuousTokenModelFamily.QWEN35
    if any(marker in haystack for marker in ("qwen2.5", "qwen2_5", "qwen2-5")) or "qwen25" in compact:
        return ContinuousTokenModelFamily.QWEN25
    if "qwen3" in compact:
        return ContinuousTokenModelFamily.QWEN3
    logger.warning(
        "No model-specific Continuous Token builder matched model_path=%r, tokenizer_name_or_path=%r; "
        "falling back to the default ContinuousTokenBuilder.",
        model_path,
        tokenizer_name_or_path or _tokenizer_name_or_path(tokenizer),
    )
    return ContinuousTokenModelFamily.DEFAULT


def create_continuous_token_builder(
    tokenizer: Any,
    *,
    model_family: str | ContinuousTokenModelFamily = "auto",
    model_path: str | None = None,
    tokenizer_name_or_path: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
    processor: Any | None = None,
    **builder_kwargs: Any,
) -> Any:
    """Instantiate the Continuous Token builder inferred from the model/tokenizer.

    The model family is inferred from ``model_path`` / tokenizer name (an explicit
    ``model_family`` is honored mainly for testing). Whether the run is text-only
    or vision-language is decided by the presence of a multimodal ``processor``.

    Resolution rules:
      * Text (no multimodal processor): use the inferred model-specific text
        builder, or the default builder when nothing matched (a warning is emitted
        by the inference step in that case).
      * VL (multimodal processor present):
          - If the name resolved to a VL family, use that VL builder.
          - If the name resolved to a *unified* text family that carries no ``vl``
            marker, upgrade it: an unrecognized model (``default``) becomes the
            default VL builder (with a warning), and ``gemma4`` becomes its VL
            builder.
          - Any other recognized text-specific family paired with a processor is
            treated as a misconfiguration and raises: a VL model's path should
            identify it as VL (resolving to a ``*_vl`` family), and a text-only
            model should not be loaded with a multimodal processor.
    """
    resolved_family = resolve_continuous_token_model_family(
        model_family,
        model_path=model_path,
        tokenizer=tokenizer,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    builder_cls = get_continuous_token_builder_class(resolved_family)
    has_mm_processor = _is_multimodal_processor(processor)

    if has_mm_processor:
        # --- Vision-language run ---
        # mm_processor_kwargs is a multimodal-only concern, so (like ``processor``) it
        # is passed only to VL builders; text builders never receive it.
        if builder_cls.supports_multimodal():
            # The name already identified a model-specific VL family.
            logger.info("Creating Continuous Token builder: family=%s class=%s", resolved_family, builder_cls)
            return builder_cls(
                tokenizer,
                processor,
                chat_template_kwargs=chat_template_kwargs,
                mm_processor_kwargs=mm_processor_kwargs,
                **builder_kwargs,
            )

        # Inferred a text family, but a multimodal processor is present. Only the
        # unified checkpoints with no ``vl`` marker are safe to auto-upgrade.
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
            f"processor was provided. If this is a vision-language model, its model path/name should "
            f"identify it as VL so it resolves to a '*_vl' family (e.g. 'Qwen2.5-VL-7B-Instruct'). "
            f"If it is a text-only model, it should not be loaded with a multimodal processor."
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


def _tokenizer_name_or_path(tokenizer: Any | None) -> str | None:
    if tokenizer is None:
        return None
    name = getattr(tokenizer, "name_or_path", None)
    if name:
        return str(name)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict) and init_kwargs.get("name_or_path"):
        return str(init_kwargs["name_or_path"])
    return None
