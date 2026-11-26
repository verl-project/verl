#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
from dataclasses import dataclass


@dataclass
class EntropyReductionEnum:
    """
    Enum for the reduction method of cross entropy.
    """

    _None = 0
    _Sum = 1
    _Mean = 2


def get_entropy_reduction_enum_number(reduction: str) -> int:
    """
    Get the enum number for the reduction method of cross entropy.
    """
    _enum = EntropyReductionEnum._None
    if reduction == "none":
        _enum = EntropyReductionEnum._None
    elif reduction == "sum":
        _enum = EntropyReductionEnum._Sum
    elif reduction == "mean":
        _enum = EntropyReductionEnum._Mean
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return _enum


def get_entropy_reduction_enum(ce_reduction: int) -> EntropyReductionEnum:
    """
    Get the enum for the reduction method of cross entropy.
    """
    _enum = EntropyReductionEnum._None
    if ce_reduction == 0:
        _enum = EntropyReductionEnum._None
    elif ce_reduction == 1:
        _enum = EntropyReductionEnum._Sum
    elif ce_reduction == 2:
        _enum = EntropyReductionEnum._Mean
    else:
        raise ValueError(f"Invalid ce_reduction: {ce_reduction}")
    return _enum


@dataclass
class BackwardEnum:
    """
    Enum for the backward method.
    """

    _Total_Fuse_MN = (
        0  # Fuse d_logits & d_hidden & d_weight, no intermediate storage, requires fp32 for d_hidden & d_weight
    )
    _Total_Separate = 1  # Store d_logits, no special requirements for d_hidden & d_weight
    _Split_Dlogits_N = 2  # split d_logits along its N dimension, aka. vocab_size
    _Split_Dlogits_M = 3  # split d_logits along its M dimension, aka. num_tokens


@dataclass
class Config:
    """Configuration for efficient entropy kernel operations.

    Args:
        _backward (BackwardEnum): Backward computation method. Defaults to BackwardEnum._Split_Dlogits_N.
        _use_triton (bool): Whether to use Triton kernels for computation. Defaults to True.
    """

    _backward: BackwardEnum = BackwardEnum._Split_Dlogits_N
    _use_triton: bool = True


_config = Config()


def set_backward_method(backward_method: BackwardEnum):
    """
    Set the backward method.
    """
    global _config
    _config._backward = backward_method
