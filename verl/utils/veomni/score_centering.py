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

"""Adapter routing VeOmni's ``chunk_logprobs_function`` to shared chunked score centering."""

from functools import wraps
from typing import Optional

import torch

from verl.trainer.ppo.score_centering import (
    current_score_centering_fused_request,
    topk_and_sampled_log_probs_from_hidden,
)


def make_veomni_score_centering_adapter(original):
    """Wrap VeOmni's ``chunk_logprobs_function`` to emit ``head_log_probs`` when requested."""

    @wraps(original)
    def adapted(
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int = 1024,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        request = current_score_centering_fused_request()
        if request is None or request.topk_ids is None:
            return original(
                hidden_states,
                weights,
                labels,
                chunk_size=chunk_size,
                ignore_index=ignore_index,
                shift_labels=shift_labels,
                temperature=temperature,
                **kwargs,
            )
        if request.head_log_probs is not None:
            raise RuntimeError("score centering expected a single fused LM-head call per engine forward")
        if shift_labels is None:
            raise ValueError("VeOmni score centering requires globally shifted labels (shift_labels)")
        log_probs, entropy, head_log_probs = topk_and_sampled_log_probs_from_hidden(
            hidden_states,
            weights,
            shift_labels,
            request.topk_ids,
            temperature=temperature,
            chunk_size=request.chunk_size,
            ignore_index=ignore_index,
        )
        request.head_log_probs = head_log_probs.contiguous()
        return log_probs, entropy

    adapted._verl_sc_adapter = True
    return adapted


def install_veomni_score_centering_adapter() -> None:
    """Install the score-centering wrapper onto ``veomni.ops.kernels.cross_entropy`` once."""
    from veomni.ops.kernels import cross_entropy

    original = cross_entropy.chunk_logprobs_function
    if not getattr(original, "_verl_sc_adapter", False):
        cross_entropy.chunk_logprobs_function = make_veomni_score_centering_adapter(original)
