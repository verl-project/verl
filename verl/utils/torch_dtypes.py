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
"""Precision type utilities adapted from Cruise.

Provides helpers to convert between string/int precision identifiers and
``torch.dtype`` objects.
"""

import torch

HALF_LIST = [16, "16", "fp16", "float16", torch.float16]
FLOAT_LIST = [32, "32", "fp32", "float32", torch.float32]
BFLOAT_LIST = ["bf16", "bfloat16", torch.bfloat16]


class PrecisionType:
    """Type of precision used.

    >>> PrecisionType.HALF == 16
    True
    >>> PrecisionType.HALF in (16, "16")
    True
    """

    HALF = "16"
    FLOAT = "32"
    FULL = "64"
    BFLOAT = "bf16"
    MIXED = "mixed"

    @staticmethod
    def supported_type(precision: str | int) -> bool:
        """Check if a precision value is supported.

        Args:
            precision: The precision identifier to check.

        Returns:
            bool: True if the precision is recognized.

        """
        return any(x == precision for x in PrecisionType)

    @staticmethod
    def supported_types() -> list[str]:
        """Return a list of all supported precision type strings.

        Returns:
            list[str]: Supported precision type values.

        """
        return [x.value for x in PrecisionType]

    @staticmethod
    def is_fp16(precision):
        """Check if precision represents float16.

        Args:
            precision: A precision identifier (int, str, or torch.dtype).

        Returns:
            bool: True if precision corresponds to float16.

        """
        return precision in HALF_LIST

    @staticmethod
    def is_fp32(precision):
        """Check if precision represents float32.

        Args:
            precision: A precision identifier (int, str, or torch.dtype).

        Returns:
            bool: True if precision corresponds to float32.

        """
        return precision in FLOAT_LIST

    @staticmethod
    def is_bf16(precision):
        """Check if precision represents bfloat16.

        Args:
            precision: A precision identifier (int, str, or torch.dtype).

        Returns:
            bool: True if precision corresponds to bfloat16.

        """
        return precision in BFLOAT_LIST

    @staticmethod
    def to_dtype(precision):
        """Convert a precision identifier to the corresponding ``torch.dtype``.

        Args:
            precision: A precision identifier (int, str, or torch.dtype).

        Returns:
            torch.dtype: The corresponding PyTorch dtype.

        Raises:
            RuntimeError: If the precision is not recognized.

        """
        if precision in HALF_LIST:
            return torch.float16
        elif precision in FLOAT_LIST:
            return torch.float32
        elif precision in BFLOAT_LIST:
            return torch.bfloat16
        else:
            raise RuntimeError(f"unexpected precision: {precision}")

    @staticmethod
    def to_str(precision):
        """Convert a ``torch.dtype`` to its short string representation.

        Args:
            precision: A ``torch.dtype`` value.

        Returns:
            str: One of ``"fp16"``, ``"fp32"``, or ``"bf16"``.

        Raises:
            RuntimeError: If the precision is not recognized.

        """
        if precision == torch.float16:
            return "fp16"
        elif precision == torch.float32:
            return "fp32"
        elif precision == torch.bfloat16:
            return "bf16"
        else:
            raise RuntimeError(f"unexpected precision: {precision}")
