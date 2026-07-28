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

"""Unit tests for the ``_keep_in_fp32_modules`` helpers used by FSDP2 (verl#7092).

These cover the pure matching / casting / unit-selection logic on CPU; the
end-to-end FSDP2 behaviour lives in
``tests/special_distributed/test_fsdp2_keep_in_fp32.py``.
"""

import re

import pytest
import torch
import torch.nn as nn
import transformers
from packaging.version import Version

from verl.utils import fsdp_utils
from verl.utils.fsdp_utils import (
    _defines_forward,
    _keep_in_fp32_regex,
    _select_keep_in_fp32_wrap_targets,
    cast_module_to_dtype_keeping_fp32_modules,
    get_keep_in_fp32_module_names,
)


# The fixtures define real `forward` methods on purpose: a module that never
# runs cannot be an FSDP2 unit, so a parameter-holding module without forward
# would not be a realistic stand-in for anything verl wraps.
class Leaf(nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.proj = nn.Linear(size, size, bias=False)
        self.register_buffer("scale", torch.ones(size))
        self.register_buffer("steps", torch.zeros(size, dtype=torch.int64))

    def forward(self, x):
        return self.proj(x) * self.scale.to(x.dtype)


class Wrapper(nn.Module):
    """Container with no direct parameters but a real forward."""

    def __init__(self, size=4):
        super().__init__()
        self.nested = Leaf(size)

    def forward(self, x):
        return self.nested(x)


class Block(nn.Module):
    def __init__(self, size=4):
        super().__init__()
        self.dense = nn.Linear(size, size, bias=False)
        self.sensitive = Leaf(size)
        self.wrapper = Wrapper(size)

    def forward(self, x):
        return self.dense(x) + self.sensitive(x) + self.wrapper(x)


class Model(nn.Module):
    _keep_in_fp32_modules = ["dense"]
    _keep_in_fp32_modules_strict = ["sensitive"]

    def __init__(self, size=4):
        super().__init__()
        self.embed = nn.Embedding(8, size)
        self.blocks = nn.ModuleList([Block(size) for _ in range(2)])

    def forward(self, idx):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return x


def _dtypes(model):
    return {name: state.dtype for name, state in list(model.named_parameters()) + list(model.named_buffers())}


# ---------------------------------------------------------------------------
# name resolution
# ---------------------------------------------------------------------------


def test_regular_list_is_fp16_only():
    """HF only honours the non-strict list for fp16 -- not for bf16."""
    model = Model()
    assert get_keep_in_fp32_module_names(model, torch.bfloat16) == ["sensitive"]
    assert get_keep_in_fp32_module_names(model, torch.float16) == ["dense", "sensitive"]


def test_no_low_precision_target_matches_nothing():
    model = Model()
    assert get_keep_in_fp32_module_names(model, torch.float32) == []
    assert get_keep_in_fp32_module_names(model, None) == []


def test_names_are_deduplicated_deterministically():
    model = Model()
    model._keep_in_fp32_modules_strict = ["sensitive", "dense", "sensitive"]
    model._keep_in_fp32_modules = ["dense"]
    assert get_keep_in_fp32_module_names(model, torch.float16) == ["dense", "sensitive"]


def test_only_the_top_most_declaration_is_read():
    """4.56.1 reads the top-level model; 5.10.0's post_init has already folded
    every child's declaration into it. Re-collecting from nested models would
    widen 4.56.1's behaviour."""
    model = Model()
    model.blocks[0]._keep_in_fp32_modules_strict = ["nested"]
    assert get_keep_in_fp32_module_names(model, torch.bfloat16) == ["sensitive"]


def test_declarations_are_read_through_a_wrapper():
    """verl may hand us a PEFT-style wrapper around the PreTrainedModel."""

    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.base_model = inner

    model = Model()
    assert get_keep_in_fp32_module_names(Wrapper(model), torch.bfloat16) == ["sensitive"]


def test_set_valued_declarations_resolve_deterministically():
    """Transformers 5.x stores these as `set`, whose iteration order varies."""
    model = Model()
    model._keep_in_fp32_modules_strict = {"sensitive", "dense", "wrapper"}
    model._keep_in_fp32_modules = {"dense"}
    assert get_keep_in_fp32_module_names(model, torch.float16) == ["dense", "sensitive", "wrapper"]


def test_list_and_set_declarations_agree():
    as_list, as_set = Model(), Model()
    as_list._keep_in_fp32_modules_strict = ["wrapper", "sensitive"]
    as_set._keep_in_fp32_modules_strict = {"wrapper", "sensitive"}
    assert get_keep_in_fp32_module_names(as_list, torch.bfloat16) == get_keep_in_fp32_module_names(
        as_set, torch.bfloat16
    )


def test_glob_declarations_are_parsed_not_rejected():
    """A glob is legal in 5.x; resolution must never reject the declaration."""
    model = Model()
    model._keep_in_fp32_modules_strict = ["sensitive", "block*"]
    assert get_keep_in_fp32_module_names(model, torch.bfloat16) == ["block*", "sensitive"]
    assert get_keep_in_fp32_module_names(model, torch.float32) == []


def test_model_without_keep_lists_resolves_empty():
    assert get_keep_in_fp32_module_names(Block(), torch.bfloat16) == []


# ---------------------------------------------------------------------------
# matching semantics
# ---------------------------------------------------------------------------


# Oracles derived from the upstream sources, not from this implementation:
#   4.56.1 modeling_utils.py:5127
#       re.compile("|".join(rf"((^|\.){m}($|\.))" for m in keep_in_fp32_modules))
#   5.10.0 core_model_loading.build_glob_alternation
#       branches.append(f"(?P<{group}>{glob.replace('*', '.*')})"); re.compile("|".join(branches))
def _oracle_4x(names):
    return re.compile("|".join(rf"((^|\.){name}($|\.))" for name in names))


def _oracle_5x(names):
    branches = [f"(?P<g{i}>{name.replace('*', '.*')})" for i, name in enumerate(names)]
    return re.compile("|".join(branches))


SEGMENT_CASES = [
    ("sensitive.proj.weight", True),  # leading segment
    ("blocks.0.sensitive.proj.weight", True),  # nested segment
    ("blocks.0.sensitive", True),  # trailing segment
    ("blocks.0.dense.weight", False),
]
# these differ between the two versions: substrings of a segment
SUBSTRING_CASES = ["blocks.0.insensitive.proj.weight", "blocks.0.sensitive_extra.weight"]


@pytest.mark.parametrize("fqn,expected", SEGMENT_CASES)
def test_matcher_handles_plain_segment_names(fqn, expected, monkeypatch):
    """A plain segment name resolves identically on both versions."""
    for glob_matching in (False, True):
        monkeypatch.setattr(fsdp_utils, "_keep_in_fp32_uses_glob_matching", lambda v=glob_matching: v)
        assert bool(_keep_in_fp32_regex(["sensitive"]).search(fqn)) is expected


@pytest.mark.parametrize("fqn", SUBSTRING_CASES)
def test_substring_does_not_match_on_4x(fqn, monkeypatch):
    monkeypatch.setattr(fsdp_utils, "_keep_in_fp32_uses_glob_matching", lambda: False)
    assert _keep_in_fp32_regex(["sensitive"]).search(fqn) is None
    assert _oracle_4x(["sensitive"]).search(fqn) is None


@pytest.mark.parametrize("fqn", SUBSTRING_CASES)
def test_substring_matches_on_5x(fqn, monkeypatch):
    monkeypatch.setattr(fsdp_utils, "_keep_in_fp32_uses_glob_matching", lambda: True)
    assert _keep_in_fp32_regex(["sensitive"]).search(fqn) is not None
    assert _oracle_5x(["sensitive"]).search(fqn) is not None


@pytest.mark.parametrize(
    "fqn,expected",
    [
        ("blocks.0.dense.weight", True),  # 'block*.dense' expands across the index
        ("blocks.12.dense.weight", True),
        ("blocks.0.sensitive.proj.weight", False),  # right prefix, wrong leaf
        ("stack.0.dense.weight", False),  # wrong prefix
    ],
)
def test_glob_declarations_match_on_5x(fqn, expected, monkeypatch):
    monkeypatch.setattr(fsdp_utils, "_keep_in_fp32_uses_glob_matching", lambda: True)
    assert bool(_keep_in_fp32_regex(["block*.dense"]).search(fqn)) is expected
    assert bool(_oracle_5x(["block*.dense"]).search(fqn)) is expected


@pytest.mark.parametrize("glob_matching", [False, True])
@pytest.mark.parametrize("fqn", [*[c[0] for c in SEGMENT_CASES], *SUBSTRING_CASES])
def test_matcher_agrees_with_the_upstream_oracle(glob_matching, fqn, monkeypatch):
    """Full agreement with the pattern each version builds internally."""
    monkeypatch.setattr(fsdp_utils, "_keep_in_fp32_uses_glob_matching", lambda: glob_matching)
    oracle = _oracle_5x if glob_matching else _oracle_4x
    names = ["sensitive", "wrapper"]
    assert bool(_keep_in_fp32_regex(names).search(fqn)) is bool(oracle(names).search(fqn))


def test_installed_transformers_selects_its_native_matcher():
    uses_glob_matching = Version(transformers.__version__).major >= 5
    assert fsdp_utils._keep_in_fp32_uses_glob_matching() is uses_glob_matching


def test_regex_is_none_without_names():
    assert _keep_in_fp32_regex([]) is None


# ---------------------------------------------------------------------------
# dtype casting
# ---------------------------------------------------------------------------


def test_cast_keeps_matched_params_and_float_buffers_in_fp32():
    model = Model()
    kept = cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16)

    dtypes = _dtypes(model)
    for layer in range(2):
        assert dtypes[f"blocks.{layer}.sensitive.proj.weight"] == torch.float32
        assert dtypes[f"blocks.{layer}.sensitive.scale"] == torch.float32
        # non-keep control module follows the target dtype
        assert dtypes[f"blocks.{layer}.dense.weight"] == torch.bfloat16
        assert dtypes[f"blocks.{layer}.wrapper.nested.proj.weight"] == torch.bfloat16
        # integer buffers are never touched
        assert dtypes[f"blocks.{layer}.sensitive.steps"] == torch.int64
    assert dtypes["embed.weight"] == torch.bfloat16
    assert sorted(kept) == [
        "blocks.0.sensitive.proj.weight",
        "blocks.0.sensitive.scale",
        "blocks.1.sensitive.proj.weight",
        "blocks.1.sensitive.scale",
    ]


def test_cast_under_fp16_keeps_the_union_of_both_lists():
    model = Model()
    cast_module_to_dtype_keeping_fp32_modules(model, torch.float16)
    dtypes = _dtypes(model)
    assert dtypes["blocks.0.dense.weight"] == torch.float32
    assert dtypes["blocks.0.sensitive.proj.weight"] == torch.float32
    assert dtypes["blocks.0.wrapper.nested.proj.weight"] == torch.float16


def test_cast_preserves_values_of_kept_modules():
    model = Model()
    before = model.blocks[0].sensitive.proj.weight.detach().clone()
    control = model.blocks[0].dense.weight.detach().clone()
    cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16)
    assert torch.equal(model.blocks[0].sensitive.proj.weight, before)
    assert torch.equal(model.blocks[0].dense.weight.float(), control.to(torch.bfloat16).float())


def test_cast_without_keep_lists_matches_plain_to():
    model, reference = Block(), Block()
    reference.load_state_dict(model.state_dict())
    assert cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16) == []
    reference.to(torch.bfloat16)
    assert _dtypes(model) == _dtypes(reference)


def _tie(model, layer, target_attr):
    """Tie ``sensitive.proj.weight`` to another parameter, returning both FQNs."""
    block = model.blocks[layer]
    holder = block if target_attr == "dense" else block.sensitive
    getattr(holder, target_attr).weight = block.sensitive.proj.weight
    prefix = f"blocks.{layer}"
    other = f"{prefix}.dense.weight" if target_attr == "dense" else f"{prefix}.sensitive.{target_attr}.weight"
    return f"{prefix}.sensitive.proj.weight", other


def test_cast_keeps_tied_group_when_every_alias_matches():
    model = Model()
    model.blocks[0].sensitive.proj2 = nn.Linear(4, 4, bias=False)
    matched, also_matched = _tie(model, 0, "proj2")

    kept = cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16)

    assert matched in kept and also_matched in kept
    assert model.blocks[0].sensitive.proj.weight.dtype == torch.float32
    assert model.blocks[0].sensitive.proj2.weight is model.blocks[0].sensitive.proj.weight


def test_cast_keeps_tied_group_when_only_one_alias_matches():
    """The aliases are one tensor: a single matching name keeps the whole group.

    Deciding per name would depend on which alias ``named_parameters`` yields
    first, and would silently downcast a weight a keep rule explicitly named.
    """
    model = Model()
    matched, unmatched = _tie(model, 0, "dense")
    assert not _keep_in_fp32_regex(["sensitive"]).search(unmatched)

    kept = cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16)

    assert matched in kept and unmatched in kept
    assert model.blocks[0].dense.weight.dtype == torch.float32
    assert model.blocks[0].sensitive.proj.weight is model.blocks[0].dense.weight
    # untied siblings are unaffected
    assert model.blocks[1].sensitive.proj.weight.dtype == torch.float32
    assert model.blocks[1].dense.weight.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# FSDP2 unit selection
# ---------------------------------------------------------------------------


def _names(model, modules):
    lookup = {id(m): n for n, m in model.named_modules()}
    return [lookup[id(m)] for m in modules]


def test_wrap_targets_are_the_matched_modules():
    model = Model()
    targets = _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16))
    assert targets == ["blocks.0.sensitive", "blocks.1.sensitive"]


def test_partially_matched_ancestors_are_skipped():
    """A unit must be all-or-nothing: FSDP2 rejects mixed original dtypes."""
    model = Model()
    targets = _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16))
    # 'blocks' / 'blocks.0' also hold unmatched parameters (dense, wrapper)
    assert "blocks" not in targets
    assert "blocks.0" not in targets


def test_wrap_targets_collapse_parent_child_overlap_and_duplicates():
    model = Model()
    model._keep_in_fp32_modules_strict = ["sensitive", "proj", "sensitive"]
    targets = _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16))
    # 'sensitive' is listed twice and 'proj' is its child -- one unit results.
    # 'proj' also matches inside 'wrapper.nested'; since that is 'wrapper''s only
    # parameter, the topmost fully-matched module ('wrapper') becomes the unit.
    assert targets == [
        "blocks.0.sensitive",
        "blocks.0.wrapper",
        "blocks.1.sensitive",
        "blocks.1.wrapper",
    ]


def test_fully_matched_ancestor_collapses_to_one_unit():
    """A wrappable ancestor whose whole subtree matches becomes a single unit."""
    model = Model()
    model._keep_in_fp32_modules_strict = ["wrapper"]
    model._keep_in_fp32_modules = None
    # 'wrapper' has no direct parameters, yet it owns every matched one below it
    assert _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)) == [
        "blocks.0.wrapper",
        "blocks.1.wrapper",
    ]
    assert "blocks.0.wrapper.nested" not in _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16))


def test_module_list_descends_to_its_children():
    """`fully_shard` rejects ModuleList outright, so select the children.

    Selecting the container would raise inside the wrapping loop, after earlier
    units had already been wrapped.
    """
    model = Model()
    model._keep_in_fp32_modules_strict = ["blocks"]
    model._keep_in_fp32_modules = None
    # every parameter lives under the ModuleList, so its children carry the policy
    assert _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)) == ["blocks.0", "blocks.1"]


def test_module_dict_descends_to_its_children():
    model = Model()
    model.bag = nn.ModuleDict({"leaf": Leaf()})
    model._keep_in_fp32_modules_strict = ["bag"]
    model._keep_in_fp32_modules = None
    assert _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)) == ["bag.leaf"]


def test_plain_container_without_forward_descends_to_its_child():
    """A namespacing `nn.Module` is accepted by fully_shard but never called.

    Its forward hook would never fire, so its parameters would stay sharded
    during compute. The callable child has to be the unit instead.
    """
    model = Model()
    model.holder = nn.Module()
    model.holder.child = Leaf()
    model._keep_in_fp32_modules_strict = ["holder"]
    model._keep_in_fp32_modules = None

    targets = _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16))
    assert targets == ["holder.child"], targets
    assert not _defines_forward(model.holder)
    assert _defines_forward(model.holder.child.proj)


def test_container_with_custom_forward_is_still_a_unit():
    """A parameter-less container that implements forward stays eligible."""

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Leaf()

        def forward(self, x):
            return self.inner(x)

    model = Model()
    model.gate = Gate()
    model._keep_in_fp32_modules_strict = ["gate"]
    model._keep_in_fp32_modules = None

    assert _defines_forward(model.gate)
    assert _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)) == ["gate"]


@pytest.mark.parametrize("container", ["ParameterList", "ParameterDict"])
def test_parameter_containers_fail_closed(container):
    """They define no forward and own parameters directly: no unit can hold them."""
    model = Model()
    if container == "ParameterList":
        model.bag = nn.ParameterList([nn.Parameter(torch.ones(4))])
    else:
        model.bag = nn.ParameterDict({"scale": nn.Parameter(torch.ones(4))})
    model._keep_in_fp32_modules_strict = ["bag"]
    model._keep_in_fp32_modules = None

    assert not _defines_forward(model.bag)
    with pytest.raises(ValueError, match="no fp32-keep unit owns it"):
        _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)


def test_parameter_container_failure_leaves_no_partially_wrapped_model():
    from torch.distributed.fsdp import FSDPModule

    model = Model()
    model.bag = nn.ParameterList([nn.Parameter(torch.ones(4))])
    model._keep_in_fp32_modules_strict = ["bag"]
    model._keep_in_fp32_modules = None

    with pytest.raises(ValueError):
        _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)
    assert not [name for name, mod in model.named_modules() if isinstance(mod, FSDPModule)]


def test_parameterless_module_is_never_a_target():
    model = Model()
    model.blocks[0].empty = nn.Module()
    model._keep_in_fp32_modules_strict = ["empty"]
    model._keep_in_fp32_modules = None
    assert _select_keep_in_fp32_wrap_targets(model, torch.bfloat16) == []


def test_tied_group_inside_one_unit_is_isolated():
    """All aliases under a single candidate unit -> that unit owns the tensor."""
    model = Model()
    model.blocks[0].sensitive.proj2 = nn.Linear(4, 4, bias=False)
    _tie(model, 0, "proj2")
    targets = _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16))
    assert targets == ["blocks.0.sensitive", "blocks.1.sensitive"]


def test_tied_group_matched_via_one_alias_is_isolated_at_the_covering_unit():
    """Only 'proj' matches, but the tied sibling sits in the same unit."""
    model = Model()
    model.blocks[0].sensitive.other = nn.Linear(4, 4, bias=False)
    model.blocks[0].sensitive.other.weight = model.blocks[0].sensitive.proj.weight
    model._keep_in_fp32_modules_strict = ["proj"]
    model._keep_in_fp32_modules = None

    targets = _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16))
    # 'sensitive' covers both aliases; 'wrapper' covers the untied block-1 match
    assert "blocks.0.sensitive" in targets
    assert "blocks.0.sensitive.proj" not in targets


def test_tie_across_candidate_units_fails_closed():
    """FSDP2 gives a parameter to exactly one unit -- refuse before wrapping."""
    model = Model()
    matched, unmatched = _tie(model, 0, "dense")
    model._keep_in_fp32_modules_strict = ["sensitive", "dense"]
    model._keep_in_fp32_modules = None

    with pytest.raises(ValueError, match="tied across FSDP2 unit boundaries") as excinfo:
        _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)
    message = str(excinfo.value)
    assert matched in message and unmatched in message
    assert "tie_word_embeddings" in message


def test_tie_reaching_outside_every_unit_fails_closed():
    """One alias matches, its sibling lives in a module that is not a unit."""
    model = Model()
    matched, unmatched = _tie(model, 0, "dense")

    with pytest.raises(ValueError, match="tied across FSDP2 unit boundaries"):
        _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)


# ---------------------------------------------------------------------------
# mixed parameter dtypes inside a matched unit (adapter-style injection)
# ---------------------------------------------------------------------------


def _inject_adapter(module, dtype):
    """A local LoRA-shaped adapter. Deliberately not peft: no extra dependency."""
    module.lora_A = nn.Linear(4, 2, bias=False)
    module.lora_B = nn.Linear(2, 4, bias=False)
    for adapter in (module.lora_A, module.lora_B):
        adapter.weight.data = adapter.weight.data.to(dtype)
    return module


def test_all_fp32_keep_unit_is_isolated_with_adapter():
    """An adapter that stays fp32 keeps the unit uniform, so isolation works."""
    model = Model()
    cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16)
    _inject_adapter(model.blocks[0].sensitive, torch.float32)

    targets = _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16))
    assert targets == ["blocks.0.sensitive", "blocks.1.sensitive"]


def test_mixed_dtype_keep_unit_fails_closed():
    """A bf16 adapter inside an fp32-keep module cannot be expressed."""
    model = Model()
    cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16)
    _inject_adapter(model.blocks[0].sensitive, torch.bfloat16)

    with pytest.raises(ValueError) as excinfo:
        _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)

    message = str(excinfo.value)
    assert "blocks.0.sensitive" in message  # matched module FQN
    assert "blocks.0.sensitive.proj.weight" in message  # the fp32 keep parameter
    assert "blocks.0.sensitive.lora_A.weight" in message  # the offending parameter
    assert "torch.bfloat16" in message  # and its dtype
    assert "all-gathers all of its parameters at one dtype" in message


def test_mixed_dtype_failure_leaves_no_partially_wrapped_model():
    """The check must run before the first fully_shard call."""
    from torch.distributed.fsdp import FSDPModule

    model = Model()
    cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16)
    _inject_adapter(model.blocks[0].sensitive, torch.bfloat16)

    with pytest.raises(ValueError):
        _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)
    assert not [name for name, mod in model.named_modules() if isinstance(mod, FSDPModule)]


def test_mixed_dtype_without_keep_list_is_untouched():
    """The same mixed-dtype model is unaffected when nothing declares fp32-keep."""
    model = Model()
    model._keep_in_fp32_modules = None
    model._keep_in_fp32_modules_strict = None
    cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16)
    _inject_adapter(model.blocks[0].sensitive, torch.bfloat16)

    assert _select_keep_in_fp32_wrap_targets(model, torch.bfloat16) == []
    assert model.blocks[0].sensitive.proj.weight.dtype == torch.bfloat16


def test_unmatched_module_with_mixed_dtypes_is_not_an_error():
    """Only a *matched* unit is a contract violation; others just are not units."""
    model = Model()
    cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16)
    _inject_adapter(model.blocks[0].dense, torch.bfloat16)

    targets = _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16))
    assert targets == ["blocks.0.sensitive", "blocks.1.sensitive"]


# ---------------------------------------------------------------------------
# every keep parameter must end up owned by exactly one unit
# ---------------------------------------------------------------------------


def test_root_level_keep_parameter_fails_closed():
    """A parameter hanging off the root can never become its own unit.

    Leaving it would keep it in the bf16 root unit -- a silent contract breach.
    """
    model = Model()
    model.logit_scale = nn.Parameter(torch.ones(1))
    model._keep_in_fp32_modules_strict = ["logit_scale"]
    model._keep_in_fp32_modules = None

    with pytest.raises(ValueError, match="no fp32-keep unit owns it") as excinfo:
        _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)
    message = str(excinfo.value)
    assert "logit_scale" in message
    assert "torch.bfloat16" in message  # the dtype it would have been degraded to


def test_keep_module_already_degraded_fails_closed():
    """All matched parameters already low precision => preservation was bypassed."""
    model = Model()
    model.to(torch.bfloat16)  # the blanket cast this fix exists to replace

    with pytest.raises(ValueError, match="already in a lower precision") as excinfo:
        _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)
    message = str(excinfo.value)
    assert "blocks.0.sensitive" in message
    assert "torch.bfloat16" in message
    assert "cast_module_to_dtype_keeping_fp32_modules" in message


def test_owner_failures_leave_no_partially_wrapped_model():
    from torch.distributed.fsdp import FSDPModule

    for build in (
        lambda: _root_keep_model(),
        lambda: _degraded_keep_model(),
    ):
        model = build()
        with pytest.raises(ValueError):
            _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)
        assert not [name for name, mod in model.named_modules() if isinstance(mod, FSDPModule)]


def _root_keep_model():
    model = Model()
    model.logit_scale = nn.Parameter(torch.ones(1))
    model._keep_in_fp32_modules_strict = ["logit_scale"]
    model._keep_in_fp32_modules = None
    return model


def _degraded_keep_model():
    model = Model()
    model.to(torch.bfloat16)
    return model


def test_child_owned_keep_parameters_still_resolve():
    """The healthy shape is unaffected by the ownership checks."""
    model = Model()
    cast_module_to_dtype_keeping_fp32_modules(model, torch.bfloat16)
    assert _names(model, _select_keep_in_fp32_wrap_targets(model, torch.bfloat16)) == [
        "blocks.0.sensitive",
        "blocks.1.sensitive",
    ]


def test_no_targets_without_keep_lists_or_low_precision():
    assert _select_keep_in_fp32_wrap_targets(Block(), torch.bfloat16) == []
    assert _select_keep_in_fp32_wrap_targets(Model(), torch.float32) == []
    assert _select_keep_in_fp32_wrap_targets(Model(), None) == []
