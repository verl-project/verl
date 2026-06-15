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
from typing import Any

from verl.utils.continuous_token import (
    ContinuousTokenBuilder,
    GLMContinuousTokenBuilder,
    MiniMaxContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
)

logger = logging.getLogger(__name__)

_CONTINUOUS_TOKEN_BUILDER_REGISTRY: dict[str, type[Any]] = {
    "default": ContinuousTokenBuilder,
    "qwen": QwenContinuousTokenBuilder,
    "qwen25": QwenContinuousTokenBuilder,
    "qwen3": QwenContinuousTokenBuilder,
    "qwen35": QwenContinuousTokenBuilder,
    "minimax": MiniMaxContinuousTokenBuilder,
    "minimaxm2": MiniMaxContinuousTokenBuilder,
    "minimaxm25": MiniMaxContinuousTokenBuilder,
    "minimaxm27": MiniMaxContinuousTokenBuilder,
    "glm47": GLMContinuousTokenBuilder,
    "glm5": GLMContinuousTokenBuilder,
}

CONTINUOUS_TOKEN_BUILDER_FAMILIES = tuple(_CONTINUOUS_TOKEN_BUILDER_REGISTRY)


def get_continuous_token_builder_class(model_family: str) -> type[Any]:
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
    model_family: str,
    *,
    model_path: str | None = None,
    tokenizer: Any | None = None,
    tokenizer_name_or_path: str | None = None,
) -> str:
    """Resolve ``auto`` to a concrete family, or canonicalize an explicit family."""
    family = _normalize_model_family(model_family)
    if family != "auto":
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
) -> str:
    """Infer a built-in model family from model/tokenizer names.

    Unknown models intentionally fall back to ``default`` so enabling
    ``model_family=auto`` remains conservative.
    """
    candidates = [model_path, tokenizer_name_or_path, _tokenizer_name_or_path(tokenizer)]
    haystack = " ".join(str(item).lower() for item in candidates if item)
    compact = re.sub(r"[^a-z0-9]+", "", haystack)

    if any(marker in haystack for marker in ("glm-5", "glm_5")) or "glm5" in compact:
        return "glm5"
    if any(marker in haystack for marker in ("glm-4.7", "glm_4.7", "glm4.7")) or "glm47" in compact:
        return "glm47"
    if "minimaxm27" in compact:
        return "minimaxm27"
    if "minimaxm25" in compact:
        return "minimaxm25"
    if "minimaxm2" in compact:
        return "minimaxm2"
    if "minimax" in compact:
        return "minimax"
    if any(marker in haystack for marker in ("qwen3.5", "qwen3_5", "qwen3-5")) or "qwen35" in compact:
        return "qwen35"
    if any(marker in haystack for marker in ("qwen2.5", "qwen2_5", "qwen2-5")) or "qwen25" in compact:
        return "qwen25"
    if "qwen3" in compact:
        return "qwen3"
    logger.warning(
        "No model-specific Continuous Token builder matched model_path=%r, tokenizer_name_or_path=%r; "
        "falling back to the default ContinuousTokenBuilder.",
        model_path,
        tokenizer_name_or_path or _tokenizer_name_or_path(tokenizer),
    )
    return "default"


def create_continuous_token_builder(
    tokenizer: Any,
    *,
    model_family: str,
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


def _normalize_model_family(model_family: str) -> str:
    if not isinstance(model_family, str) or not model_family:
        raise ValueError("Continuous Token model_family must be a non-empty string")
    family = model_family.strip().lower()
    if not family:
        raise ValueError("Continuous Token model_family must be a non-empty string")
    return re.sub(r"[^a-z0-9]+", "", family)


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
