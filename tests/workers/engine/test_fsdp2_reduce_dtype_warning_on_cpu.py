# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""A reduce dtype that differs from the parameter dtype keeps gradients unsharded.

FSDP2's `to_accumulated_grad_if_needed` only returns early when the gradient is already in
`reduce_dtype`, so bf16 params with an fp32 reduce dtype leave a whole-model fp32
`unsharded_accumulated_grad` alive for the gradient-accumulation window. That term scales with
parameter count and not with world size, so it does not respond to the per-GPU token budget —
easy to read as an activation problem. This pins that the cost is at least announced.
"""

import logging

import pytest
import torch

from verl.workers.engine.fsdp import transformer_impl


@pytest.fixture(autouse=True)
def _reset_warned_flag():
    transformer_impl._warned_reduce_dtype_grad_memory = False
    yield
    transformer_impl._warned_reduce_dtype_grad_memory = False


def test_warns_when_reduce_dtype_differs(caplog):
    with caplog.at_level(logging.WARNING, logger=transformer_impl.logger.name):
        transformer_impl._warn_if_reduce_dtype_retains_unsharded_grads(torch.bfloat16, torch.float32)

    assert len(caplog.records) == 1
    text = caplog.records[0].getMessage()
    assert "unsharded" in text
    assert "independent of world size" in text


@pytest.mark.parametrize(
    "param_dtype,reduce_dtype",
    [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32), (torch.bfloat16, None)],
)
def test_silent_when_no_upcast_happens(caplog, param_dtype, reduce_dtype):
    """Matching dtypes hit the early return, so there is no retained gradient to warn about."""
    with caplog.at_level(logging.WARNING, logger=transformer_impl.logger.name):
        transformer_impl._warn_if_reduce_dtype_retains_unsharded_grads(param_dtype, reduce_dtype)

    assert caplog.records == []


def test_warns_only_once(caplog):
    """This sits on a per-module build path, so an un-deduplicated warning would repeat per layer."""
    with caplog.at_level(logging.WARNING, logger=transformer_impl.logger.name):
        for _ in range(5):
            transformer_impl._warn_if_reduce_dtype_retains_unsharded_grads(torch.bfloat16, torch.float32)

    assert len(caplog.records) == 1
