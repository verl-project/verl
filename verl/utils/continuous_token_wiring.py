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
"""Continuous Token builder registry and model-family resolution."""

from __future__ import annotations

import logging
import re
from typing import Any

from verl.utils.continuous_token import (
    ContinuousTokenBuilder,
    GLM47ContinuousTokenBuilder,
    MiniMaxContinuousTokenBuilder,
    Qwen35ContinuousTokenBuilder,
    Qwen3ContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
)
from verl.utils.import_utils import import_external_libs

logger = logging.getLogger(__name__)

_BUILTIN_CONTINUOUS_TOKEN_BUILDER_REGISTRY: dict[str, type[Any] | None] = {
    "default": ContinuousTokenBuilder,
    "qwen": QwenContinuousTokenBuilder,
    "qwen25": QwenContinuousTokenBuilder,
    "qwen3": Qwen3ContinuousTokenBuilder,
    "qwen35": Qwen35ContinuousTokenBuilder,
    "minimax": MiniMaxContinuousTokenBuilder,
    "glm47": GLM47ContinuousTokenBuilder,
}

CONTINUOUS_TOKEN_BUILDER_FAMILIES = tuple(_BUILTIN_CONTINUOUS_TOKEN_BUILDER_REGISTRY)
CONTINUOUS_TOKEN_MODEL_FAMILY_OPTIONS = ("auto", *CONTINUOUS_TOKEN_BUILDER_FAMILIES)

_RESERVED_MODEL_FAMILIES = frozenset({"auto"})


class ContinuousTokenBuilderRegistry:
    """Registry for Continuous Token builder classes.

    Custom builders can be registered from an imported Python module:

    .. code-block:: python

        @ContinuousTokenBuilderRegistry.register("deepseek")
        class DeepseekContinuousTokenBuilder:
            ...
    """

    _registry: dict[str, type[Any]] = {}

    @classmethod
    def register(cls, model_family: str, *, exist_ok: bool = False):
        family = _normalize_model_family(model_family)
        if family in _RESERVED_MODEL_FAMILIES:
            raise ValueError(f"{family!r} is reserved and cannot be registered as a Continuous Token builder family")

        def decorator(builder_cls: type[Any]) -> type[Any]:
            if not isinstance(builder_cls, type):
                raise TypeError(f"builder_cls must be a class, got {type(builder_cls)!r}")

            existing = cls._registry.get(family)
            if existing is not None and existing is not builder_cls and not exist_ok:
                raise ValueError(
                    f"Continuous Token builder family {family!r} is already registered with "
                    f"{existing}; pass exist_ok=True to replace it"
                )
            cls._registry[family] = builder_cls
            return builder_cls

        return decorator

    @classmethod
    def get(cls, model_family: str) -> type[Any]:
        family = _normalize_model_family(model_family)
        if family in _BUILTIN_CONTINUOUS_TOKEN_BUILDER_REGISTRY:
            builder_cls = _BUILTIN_CONTINUOUS_TOKEN_BUILDER_REGISTRY[family]
            if builder_cls is None:
                raise ValueError(
                    f"Continuous Token builder family {family!r} is part of the built-in config surface, "
                    "but its builder class has not been implemented yet."
                )
            return builder_cls
        try:
            return cls._registry[family]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Continuous Token builder family {family!r}. "
                f"Registered families: {sorted(cls._registry)}. "
                f"Built-in families planned by config surface: {CONTINUOUS_TOKEN_BUILDER_FAMILIES}."
            ) from exc

    @classmethod
    def registered_families(cls) -> tuple[str, ...]:
        return tuple(sorted((*_BUILTIN_CONTINUOUS_TOKEN_BUILDER_REGISTRY, *cls._registry)))


def register_continuous_token_builder(
    model_family: str,
    builder_cls: type[Any] | None = None,
    *,
    exist_ok: bool = False,
):
    """Register a Continuous Token builder class for a model family.

    This wrapper supports both direct calls and decorator usage. New code should
    prefer ``ContinuousTokenBuilderRegistry.register`` to match Verl registries.
    """
    decorator = ContinuousTokenBuilderRegistry.register(model_family, exist_ok=exist_ok)
    if builder_cls is None:
        return decorator
    return decorator(builder_cls)


def get_continuous_token_builder_class(model_family: str) -> type[Any]:
    return ContinuousTokenBuilderRegistry.get(model_family)


def list_continuous_token_builder_families() -> tuple[str, ...]:
    return ContinuousTokenBuilderRegistry.registered_families()


def resolve_continuous_token_model_family(
    model_family: str,
    *,
    model_path: str | None = None,
    tokenizer: Any | None = None,
    tokenizer_name_or_path: str | None = None,
) -> str:
    """Resolve ``auto`` to a concrete family, or return an explicit family unchanged."""
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

    if any(marker in haystack for marker in ("glm-4.7", "glm_4.7", "glm4.7", "glm-5", "glm_5")) or any(
        marker in compact for marker in ("glm47", "glm5")
    ):
        return "glm47"
    if "minimax" in compact:
        return "minimax"
    if (
        any(marker in haystack for marker in ("qwen2.5", "qwen2_5", "qwen2-5", "qwen3.5", "qwen3_5", "qwen3-5"))
        or any(marker in compact for marker in ("qwen25", "qwen3", "qwen35"))
    ):
        return "qwen"
    return "default"


def create_continuous_token_builder(
    tokenizer: Any,
    *,
    model_family: str,
    model_path: str | None = None,
    tokenizer_name_or_path: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    custom_builder_module: str | list[str] | None = None,
    **builder_kwargs: Any,
) -> Any:
    """Instantiate the registered builder selected by config/model metadata."""
    import_external_libs(custom_builder_module)
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
    return model_family.lower()


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
