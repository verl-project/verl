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

from enum import StrEnum
import logging
import re
from typing import Any

from verl.utils.continuous_token import (
    ContinuousTokenBuilder,
    Gemma4ContinuousTokenBuilder,
    GLMContinuousTokenBuilder,
    GptOssContinuousTokenBuilder,
    MiniMaxContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
)

logger = logging.getLogger(__name__)


class ContinuousTokenModelFamily(StrEnum):
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


_CONTINUOUS_TOKEN_BUILDER_REGISTRY: dict[ContinuousTokenModelFamily, type[Any]] = {
    ContinuousTokenModelFamily.DEFAULT: ContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN: QwenContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN25: QwenContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN3: QwenContinuousTokenBuilder,
    ContinuousTokenModelFamily.QWEN35: QwenContinuousTokenBuilder,
    ContinuousTokenModelFamily.MINIMAX: MiniMaxContinuousTokenBuilder,
    ContinuousTokenModelFamily.MINIMAX_M2: MiniMaxContinuousTokenBuilder,
    ContinuousTokenModelFamily.MINIMAX_M25: MiniMaxContinuousTokenBuilder,
    ContinuousTokenModelFamily.MINIMAX_M27: MiniMaxContinuousTokenBuilder,
    ContinuousTokenModelFamily.GLM47: GLMContinuousTokenBuilder,
    ContinuousTokenModelFamily.GLM5: GLMContinuousTokenBuilder,
    ContinuousTokenModelFamily.GEMMA4: Gemma4ContinuousTokenBuilder,
    ContinuousTokenModelFamily.GPTOSS: GptOssContinuousTokenBuilder,
}

CONTINUOUS_TOKEN_BUILDER_FAMILIES = tuple(family.value for family in _CONTINUOUS_TOKEN_BUILDER_REGISTRY)


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
    model_family: str | ContinuousTokenModelFamily,
    model_path: str | None = None,
    tokenizer_name_or_path: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    **builder_kwargs: Any,
) -> Any:
    """Instantiate the registered builder selected by config/model metadata."""
    resolved_family = resolve_continuous_token_model_family(
        model_family,
        model_path=model_path,
        tokenizer=tokenizer,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    builder_cls = get_continuous_token_builder_class(resolved_family)
    logger.info("Creating Continuous Token builder: family=%s class=%s", resolved_family, builder_cls)
    return builder_cls(tokenizer, chat_template_kwargs=chat_template_kwargs, **builder_kwargs)


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
