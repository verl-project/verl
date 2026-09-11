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

"""Shared teardown for resources held in creation order."""

from __future__ import annotations

from typing import Protocol, TypeVar

from verl.single_controller.base.errors import ExceptionGroup


class _Closable(Protocol):
    def close(self) -> None: ...


OwnerT = TypeVar("OwnerT", bound=_Closable)


def close_owned(owners: list[OwnerT]) -> None:
    """Close in reverse order, retaining only failed owners for a later retry.

    The caller must serialize teardown and stop other writers first. A
    BaseException interrupts the pass without changing the ownership list.
    """
    errors: list[Exception] = []
    remaining: list[OwnerT] = []
    for owner in reversed(owners):
        try:
            owner.close()
        except Exception as exc:
            errors.append(exc)
            remaining.append(owner)
    owners[:] = reversed(remaining)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup("runtime failures", errors)
