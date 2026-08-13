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

import pytest
import torch

from verl.utils.megatron.router_replay_patch import (
    RouterReplay,
    RouterReplayAction,
    _patched_topk_routing_with_score_function,
)


def test_record_canonicalizes_thd_alignment_padding_before_dispatch():
    router_replay = RouterReplay()
    router_replay.set_router_replay_action(RouterReplayAction.RECORD)
    router_replay.record_padding_mask = torch.tensor([False, True])
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    _probs, routing_map = _patched_topk_routing_with_score_function(
        logits=logits,
        topk=1,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        score_function="softmax",
        expert_bias=None,
        fused=False,
        router_replay=router_replay,
        scaling_factor=1.0,
    )

    expected = torch.tensor([[2], [0]])
    assert torch.equal(router_replay.recorded_topk_idx, expected)
    assert routing_map[0, 2].item()
    assert routing_map[1, 0].item()


def test_record_padding_mask_shape_mismatch_hard_fails():
    router_replay = RouterReplay()
    router_replay.record_padding_mask = torch.tensor([True])

    try:
        router_replay.canonicalize_record_topk_indices(torch.tensor([[1], [2]]))
    except RuntimeError as exc:
        assert "padding mask does not match" in str(exc)
    else:
        raise AssertionError("expected router padding-mask shape mismatch to fail")


def test_record_padding_mask_scope_clears_state_and_propagates_forward_error():
    router_replay = RouterReplay()
    padding_mask = torch.tensor([False, True])

    with pytest.raises(RuntimeError, match="forward failed"):
        with RouterReplay.scoped_record_padding_mask(padding_mask):
            assert router_replay.record_padding_mask is padding_mask
            raise RuntimeError("forward failed")

    assert router_replay.record_padding_mask is None
