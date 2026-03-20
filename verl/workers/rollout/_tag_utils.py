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
"""Lightweight tag validation utilities for rollout release/resume.

This module has zero heavy dependencies (no torch, ray, etc.) so it can
be imported in unit tests without GPU or distributed infrastructure.
"""

from __future__ import annotations

_VALID_RELEASE_TAGS = frozenset({"kv_cache", "weights"})
_DEFAULT_RELEASE_TAGS = ("kv_cache", "weights")


def validate_release_tags(tags: list[str] | None) -> set[str]:
    """Normalize and validate release tags.

    Args:
        tags: List of tags to release, or None for the default (both).

    Returns:
        A set of validated tags.

    Raises:
        ValueError: If any tag is not in {"kv_cache", "weights"}.
    """
    if tags is None:
        return set(_DEFAULT_RELEASE_TAGS)
    tag_set = set(tags)
    if not tag_set:
        raise ValueError("release tags must not be empty; pass None to release all")
    unknown = tag_set - _VALID_RELEASE_TAGS
    if unknown:
        raise ValueError(f"Unknown release tags: {unknown!r}; expected subset of {sorted(_VALID_RELEASE_TAGS)}")
    return tag_set
