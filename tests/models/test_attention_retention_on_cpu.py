# Copyright contributors to the PEAR project
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

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from verl.models.transformers.attention_retention import retain_attention_in_checkpoints


class _CheckpointedLayer(torch.nn.Module):
    """A decoder layer under gradient checkpointing whose type is held under the given attribute name."""

    def __init__(self, type_name, layer_type):
        super().__init__()
        setattr(self, type_name, layer_type)
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)


@pytest.mark.parametrize("type_name", ["block_type", "layer_type"])  # transformers 5.16 and later, 5.12
def test_only_the_full_attention_layers_retain_their_attention(type_name):
    linear_layer = _CheckpointedLayer(type_name, "linear_attention")
    full_layer = _CheckpointedLayer(type_name, "full_attention")
    linear_checkpoint = linear_layer._gradient_checkpointing_func
    model = torch.nn.Sequential(linear_layer, full_layer)
    model.config = SimpleNamespace(_attn_implementation="sdpa")  # no FlashAttention forward to wrap

    retain_attention_in_checkpoints(model)

    assert linear_layer._gradient_checkpointing_func is linear_checkpoint
    assert "context_fn" in full_layer._gradient_checkpointing_func.keywords
