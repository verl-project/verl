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
"""CPU unit tests for fused-param grouping in the plain full-sync bucketer.

SGLang's DeepSeek-V4 loader rebuilds ``wqkv_a`` / ``compressor.wkv_gate`` /
``indexer.compressor.wkv_gate`` by ``torch.cat``-ing two separately-named tensors,
buffering the first arrival in a cache it creates inside ``load_weights`` and
asserting it empty on return::

    deepseek_v4.py:1547, in load_weights
        assert len(cache_wqkv_a_weight) == 0

Both halves must therefore reach the SAME ``load_weights`` call, i.e. the same
bucket. ``get_named_tensor_buckets`` splits by accumulated bytes, so without the
grouping any bucket boundary that lands between two members fires that assert.

The membership table is tested alongside because getting it wrong is invisible at
runtime: an unmatched member is simply bucketed ungrouped, which looks exactly
like "this model has no fused params" until the loader dies much later.
"""

import asyncio

import pytest
import torch

from verl.utils.fusion_groups import FUSION_GROUPS, fusion_key, fusion_match, group_size

L = "model.layers.7.self_attn"


# --------------------------------------------------------------------------- table


def test_table_covers_every_family_in_both_attention_spellings():
    keys = [k for k, _ in FUSION_GROUPS]
    for base in (
        "wqkv_a",
        "wqkv_a_scale",
        "wqkv_a_scale_inv",
        "compressor_wkv_gate",
        "indexer_compressor_wkv_gate",
    ):
        assert base in keys, f"{base} missing"
        assert base + "@attn" in keys, f"{base}@attn missing"
    assert all(group_size(k) == 2 for k, _ in FUSION_GROUPS)


@pytest.mark.parametrize(
    "name",
    [
        f"{L}.wq_a.weight",
        f"{L}.wkv.weight",
        # a bare ".wkv.weight" suffix would also match these two, which is why the
        # table spells the attention block out
        f"{L}.compressor.wkv.weight",
        f"{L}.indexer.compressor.wkv.weight",
        # Megatron-Bridge's DSv4 mapping writes layers.N.attn.*, SGLang sees self_attn.*
        "layers.7.attn.wq_a.weight",
        "layers.7.attn.compressor.wgate.weight",
        # the fp8 scale is named .scale in the export, weight_scale_inv after a
        # bf16-master requantize -- both must resolve, and to DIFFERENT groups
        "layers.0.attn.wq_a.scale",
        f"{L}.wq_a.weight_scale_inv",
    ],
)
def test_every_member_matches_exactly_one_group(name):
    assert fusion_match(name) is not None  # asserts internally on a double match


def test_weight_and_scale_are_separate_groups():
    # SGLang keys its cache on the destination param name, so wqkv_a.weight and
    # wqkv_a.weight_scale_inv are distinct entries each needing its own pair.
    assert fusion_key("layers.0.attn.wq_a.scale") == fusion_key("layers.0.attn.wkv.scale")
    assert fusion_key("layers.0.attn.wq_a.weight") == fusion_key("layers.0.attn.wkv.weight")
    assert fusion_key("layers.0.attn.wq_a.scale") != fusion_key("layers.0.attn.wq_a.weight")


def test_non_member_returns_none():
    assert fusion_key("model.layers.7.mlp.gate_proj.weight") is None
    assert fusion_key("model.embed_tokens.weight") is None


# ------------------------------------------------------------------- full-sync bucketer


def _run_buckets(items, cap):
    from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

    async def go():
        return [b async for b in get_named_tensor_buckets(iter(items), cap)]

    return asyncio.run(go())


def _t(nbytes):
    return torch.empty(nbytes, dtype=torch.uint8)


def test_full_sync_bucketer_keeps_a_group_together():
    out = _run_buckets(
        [("filler", _t(1024 - 16)), (f"{L}.wq_a.weight", _t(64)), (f"{L}.wkv.weight", _t(64))], 1024
    )
    names = [[n for n, _ in b] for b in out]
    assert len(out) == 2, "expected a real split, otherwise this proves nothing"
    assert any(f"{L}.wq_a.weight" in b and f"{L}.wkv.weight" in b for b in names)


def test_full_sync_bucketer_reunites_members_that_arrive_apart():
    out = _run_buckets(
        [(f"{L}.wq_a.weight", _t(64)), ("other", _t(64)), (f"{L}.wkv.weight", _t(64))], 1024
    )
    names = [[n for n, _ in b] for b in out]
    assert any(f"{L}.wq_a.weight" in b and f"{L}.wkv.weight" in b for b in names)


def test_full_sync_bucketer_leaves_non_members_alone():
    items = [(f"p{i}", _t(600)) for i in range(4)]
    out = _run_buckets(items, 1024)
    assert [n for b in out for n, _ in b] == [f"p{i}" for i in range(4)]
    assert len(out) == 4


def test_full_sync_bucketer_fails_loudly_on_an_incomplete_group():
    with pytest.raises(AssertionError, match="never completed"):
        _run_buckets([(f"{L}.wq_a.weight", _t(64)), ("x", _t(64))], 1024)
