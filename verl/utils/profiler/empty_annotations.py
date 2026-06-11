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

from typing import Callable, Optional


def mark_start_range(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    """Start a profiling range marker (no-op placeholder).

    Args:
        message: Message to associate with the range marker.
        color: Color for the marker visualization.
        domain: Domain for the marker.
        category: Category for the marker.

    """
    pass


def mark_end_range(range_id: str) -> None:
    """End a profiling range marker (no-op placeholder).

    Args:
        range_id: Identifier of the range to end.

    """
    pass


def mark_annotate(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> Callable:
    """Decorate a function with profiling annotation (no-op placeholder).

    Args:
        message: Message to associate with the annotation.
        color: Color for the marker visualization.
        domain: Domain for the marker.
        category: Category for the marker.

    Returns:
        A decorator that returns the original function unchanged.

    """

    def decorator(func):
        return func

    return decorator
