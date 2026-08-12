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
"""Destination params a rollout engine rebuilds from several checkpoint tensors.

SGLang's DeepSeek-V4 loader is the reason this exists. It rebuilds ``wqkv_a``,
``compressor.wkv_gate`` and ``indexer.compressor.wkv_gate`` by ``torch.cat``-ing
two separately-named tensors, buffering whichever half arrives first in a cache
it creates *inside* ``load_weights`` and asserts empty on return::

    assert len(cache_wqkv_a_weight) == 0, cache_wqkv_a_weight.keys()

So both halves must reach the SAME ``load_weights`` call. Every weight-transfer
path in verl splits by accumulated bytes with no idea which tensors belong
together, and each split point can break this independently:

* ``sglang_rollout.get_named_tensor_buckets`` -- the plain full-sync path, used
  by the NCCL checkpoint engine and by ``separate_async``'s hybrid replicas.

Today that is the only such site, but any future splitter (a sparse/delta wire,
a receiver-side re-chunker) has the same constraint, so the membership table
lives here as a single source of truth rather than inline in the bucketer. It
deliberately imports nothing.

Note this is a property of the *loader*, not of DSv4: 121 of SGLang's 190 model
files write fused params per-slice via ``stacked_params_mapping`` (a shard_id
plus ``narrow``) and need none of this, and vLLM's own DSv4 uses
``MergedColumnParallelLinear(disable_tp=True)`` and never cats at all. DSv4 on
SGLang is the only model in the tree that buffers-and-cats.
"""

# Both spellings of the attention block are carried: names reaching SGLang use
# ``self_attn`` (its loader's cache key ``model.layers.N.self_attn.wqkv_a.weight``
# back-derives an incoming ``...self_attn.wq_a.weight``) while Megatron-Bridge's
# DSv4 mapping writes ``layers.N.attn.*``. They are mutually exclusive under
# ``endswith`` -- ``self_attn`` has no dot before ``attn`` -- so listing both
# cannot introduce an ambiguity, and guessing wrong would silently disable the
# grouping rather than fail.
_ATTN_SPELLINGS = (".self_attn.", ".attn.")

# fp8 splits weight and scale into separate groups because SGLang keys its cache
# on the destination param name: ``wqkv_a.weight`` and ``wqkv_a.weight_scale_inv``
# are distinct entries and each needs its own pair.
_FAMILIES = (
    ("wqkv_a", ("wq_a.weight", "wkv.weight")),
    # Two spellings of the fp8 scale, and BOTH are needed:
    #   .scale            -- what the natively-fp8 DSv4 checkpoint stores and
    #                        what the Megatron export actually streams. Measured,
    #                        not guessed: the run named them
    #                        ``layers.0.attn.wq_a.scale`` / ``...wkv.scale``.
    #   .weight_scale_inv -- what SGLang's own param is called, and what a
    #                        verl-side requantizer of a bf16 master emits
    #                        (``name + "_scale_inv"``).
    # Getting this wrong is invisible: the weight group still pairs up and only
    # the scale group silently stays unpaired, which reads as a partial fix.
    ("wqkv_a_scale", ("wq_a.scale", "wkv.scale")),
    ("wqkv_a_scale_inv", ("wq_a.weight_scale_inv", "wkv.weight_scale_inv")),
    ("compressor_wkv_gate", ("compressor.wkv.weight", "compressor.wgate.weight")),
    (
        "indexer_compressor_wkv_gate",
        ("indexer.compressor.wkv.weight", "indexer.compressor.wgate.weight"),
    ),
)

#: ``((group_key, (suffix, ...)), ...)`` -- a param whose name ends with one of a
#: group's suffixes is one member of that group.
FUSION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (key + ("" if attn == ".self_attn." else "@attn"), tuple(attn + m for m in members))
    for attn in _ATTN_SPELLINGS
    for key, members in _FAMILIES
)


def fusion_match(name: str):
    """``(group_key, suffix)`` if ``name`` is a member of a fusion group, else None.

    Asserts on a name matching two groups: a bare ``.wkv.weight`` would match the
    attention family as well as both compressor families, so the suffixes are
    spelled out far enough to disambiguate. If that ever stops being true the
    grouping would silently mis-associate params, so fail loudly instead.
    """
    hits = [(key, sfx) for key, sfxs in FUSION_GROUPS for sfx in sfxs if name.endswith(sfx)]
    assert len(hits) <= 1, f"{name!r} matches multiple fusion groups: {hits}"
    return hits[0] if hits else None


def fusion_key(name: str):
    """``(prefix, group_key)`` identifying the specific group instance, else None.

    Two members share a key exactly when they rebuild the same destination param.
    """
    matched = fusion_match(name)
    if matched is None:
        return None
    key, sfx = matched
    return name[: -len(sfx)], key


def group_size(group_key: str) -> int:
    return len(next(s for k, s in FUSION_GROUPS if k == group_key))
