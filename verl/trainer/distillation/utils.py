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
"""
Contains utilities/classes for on-policy distillation 
"""

from typing import Union, Optional
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from enum import Enum

class Stage(Enum):
    """
    Stages for PPO training
    """
    OLD_LOG_PROB = "old_log_prob"
    REF_LOG_PROB = "ref_log_prob"
    ACTOR_UPDATE = "actor_update"

def get_topk_keys(stage: Union[str, Stage]):
    """TODO: Docstring for get_topk_keys"""
    if isinstance(stage, Stage):
        stage = stage.value
    return f"{stage}_topk_log_probs", f"{stage}_topk_indices"

def topk_logprobs_from_logits(logits: torch.Tensor, k: int, compute_both: bool, topk_indices: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:    
    """TODO: Docstring for topk_logprobs_from_logits"""
    logprobs = F.log_softmax(logits, dim=-1)

    needs_dedupe = False
    if compute_both:
        if topk_indices is None or topk_indices.shape[-1] == k:
            should_compute_topk = True
        elif topk_indices.shape[-1] == 2 * k:
            should_compute_topk = False
        else:
            raise ValueError(f"{topk_indices.shape=} is not expected with {k=}")
    else:
        if topk_indices is None:
            should_compute_topk = True
        elif topk_indices.shape[-1] == k:
            should_compute_topk = False
        else:
            raise ValueError(f"{topk_indices.shape=} is not expected with {k=}")
            

    topk_logprobs_ls = []
    topk_logprobs_indices_ls = []
    
    # Gather logits for provided indices.
    if topk_indices is not None:
        topk_logprobs = torch.gather(logprobs, dim=-1, index=topk_indices)
        topk_logprobs_ls.append(topk_logprobs)
        topk_logprobs_indices_ls.append(topk_indices)

    # Compute top-k logprobs.
    if should_compute_topk:
        topk_logprobs, topk_indices = torch.topk(logprobs, k=k, dim=-1)
        topk_logprobs_ls.append(topk_logprobs)
        topk_logprobs_indices_ls.append(topk_indices)

    topk_logprobs = torch.cat(topk_logprobs_ls, dim=-1)
    topk_indices = torch.cat(topk_logprobs_indices_ls, dim=-1)

    # If top-k have been provided AND new top-k have been computed, we need to deduplicate the indices and logprobs. 
    if needs_dedupe:

        # Make sure indices are sorted so that we can identify duplicates.
        topk_indices_diff = topk_indices.diff(dim=-1)
        if topk_indices_diff.lt(0).any():
            topk_indices, sort_indices = topk_indices.sort(dim=-1)
            topk_logprobs = torch.gather(topk_logprobs, dim=-1, index=sort_indices)
            topk_indices_diff = topk_indices.diff(dim=-1)

        # Find duplicate indices and set their prob to ~0.
        if topk_indices_diff.eq(0).any():
            index_diffs = torch.nn.functional.pad(topk_indices_diff, (0, 1), value=1)
            dupe_mask = index_diffs.eq(0)
            topk_logprobs[dupe_mask] = -torch.inf

    return topk_logprobs, topk_indices

def compute_topk_outputs(logits: torch.Tensor, batch: TensorDict, cu_seqlens: torch.Tensor):
    """
    TODO: Docstring for compute_topk_outputs
    """
    stage = batch["stage"]
    topk_logprobs, topk_indices = topk_logprobs_from_logits(logits=logits, k=2, compute_both=True, topk_indices=batch.get("topk_indices", None))
    topk_logprobs_key, topk_indices_key = get_topk_keys(stage)
    output = {
        topk_logprobs_key: torch.nested.nested_tensor_from_jagged(topk_logprobs.squeeze(0), cu_seqlens),
        topk_indices_key: torch.nested.nested_tensor_from_jagged(topk_indices.squeeze(0), cu_seqlens),
    }
    return output

def gather_topk_outputs(stage: Stage, output: TensorDict):
    """
    TODO: Docstring for gather_topk_outputs
    """
    topk_logprobs_key, topk_indices_key = get_topk_keys(stage)
    topk_logprobs = tu.get(output, topk_logprobs_key)
    if topk_logprobs is not None:
        return {
            topk_logprobs_key: topk_logprobs.float(),
            topk_indices_key: tu.get(output, topk_indices_key),
        }
    else:
        return {}

