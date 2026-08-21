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

"""Regression tests for https://github.com/verl-project/verl/issues/6259.

Reported symptom: merging an FSDP checkpoint trained on 64 GPUs (but not 32) produced
HuggingFace safetensors shards with "duplicate" keys silently dropped (e.g. 57 keys
removed as duplicates in one shard, 41 in another).

Root cause: ``FSDPModelMerger._merge_by_placement`` returns the rank-0 local tensor
*unmodified* for parameters with a "replicate" DTensor placement (parameters FSDP left
un-sharded, which becomes more likely to happen for some tensors the larger the FSDP
world size is). That local tensor can be a *view* that still shares underlying storage
with something else in the source checkpoint. Because ``_merge_by_placement`` never
clones it, two entirely different parameter keys in the merged state_dict can end up
aliasing the same tensor storage. ``transformers``' ``save_pretrained`` treats
same-storage tensors as tied weights and silently keeps only one of them per shard,
which is exactly the "duplicate keys removed" symptom reported in the issue.

These tests load the real ``verl/model_merger/fsdp_model_merger.py`` and
``verl/model_merger/base_model_merger.py`` source files directly via ``importlib``,
stubbing out heavyweight/version-sensitive dependencies (``torch.distributed._tensor``,
``accelerate``, ``transformers``, the rest of the ``verl`` package) that are not
available/compatible in every test environment. This keeps the tests exercising the
actual production code paths without requiring a real multi-GPU FSDP checkpoint or a
specific transformers/torch version.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_MERGER_DIR = REPO_ROOT / "verl" / "model_merger"


class _FakePlacement:
    def is_replicate(self) -> bool:
        return False

    def is_partial(self) -> bool:
        return False

    def is_shard(self) -> bool:
        return False


class _FakeReplicate(_FakePlacement):
    def is_replicate(self) -> bool:
        return True

    def __eq__(self, other):
        return isinstance(other, _FakeReplicate)

    def __hash__(self):
        return hash(_FakeReplicate)


class _FakeShard(_FakePlacement):
    def __init__(self, dim: int):
        self.dim = dim

    def is_shard(self) -> bool:
        return True

    def __eq__(self, other):
        return isinstance(other, _FakeShard) and other.dim == self.dim

    def __hash__(self):
        return hash((_FakeShard, self.dim))


class _FakeDTensor:
    """Minimal stand-in for ``torch.distributed.tensor.DTensor``.

    Only implements what ``FSDPModelMerger._load_and_merge_state_dicts`` /
    ``_merge_by_placement`` actually touch: ``_local_tensor`` and ``placements``.
    """

    def __init__(self, local_tensor: torch.Tensor, placements: tuple):
        self._local_tensor = local_tensor
        self.placements = placements


def _install_fake_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


@pytest.fixture()
def fsdp_merger_module(monkeypatch):
    """Import the real fsdp_model_merger.py with heavy deps stubbed out."""

    # torch 1.12 (used in some CI/dev environments) has neither
    # torch.distributed.tensor nor torch.distributed._tensor. Provide a fake
    # torch.distributed._tensor and make sure the torch.distributed.tensor import
    # (tried first in fsdp_model_merger.py) fails so it falls back to the fake.
    fake_tensor_mod = _install_fake_module(
        "torch.distributed._tensor",
        Placement=_FakePlacement,
        Shard=_FakeShard,
        Replicate=_FakeReplicate,
        DTensor=_FakeDTensor,
    )
    monkeypatch.setattr(torch.distributed, "_tensor", fake_tensor_mod, raising=False)
    monkeypatch.delitem(sys.modules, "torch.distributed.tensor", raising=False)

    # Stub verl / verl.model_merger / verl.model_merger.base_model_merger so the
    # relative `from .base_model_merger import BaseModelMerger` in
    # fsdp_model_merger.py resolves without pulling in transformers/accelerate/ray/etc.
    if "verl" not in sys.modules:
        _install_fake_module("verl", __path__=[str(REPO_ROOT / "verl")])
    if "verl.model_merger" not in sys.modules:
        _install_fake_module("verl.model_merger", __path__=[str(MODEL_MERGER_DIR)])

    class _FakeBaseModelMerger:
        def __init__(self, config=None):
            self.config = config

    _install_fake_module("verl.model_merger.base_model_merger", BaseModelMerger=_FakeBaseModelMerger)

    spec = importlib.util.spec_from_file_location(
        "verl.model_merger.fsdp_model_merger", MODEL_MERGER_DIR / "fsdp_model_merger.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["verl.model_merger.fsdp_model_merger"] = module
    spec.loader.exec_module(module)

    yield module

    for name in (
        "torch.distributed._tensor",
        "verl.model_merger.base_model_merger",
        "verl.model_merger.fsdp_model_merger",
    ):
        sys.modules.pop(name, None)


def test_merge_by_placement_replicate_does_not_alias_unrelated_keys(fsdp_merger_module):
    """Direct unit test of the buggy function.

    Simulates two *different* parameters ("param_a", "param_b") whose rank-0 local
    tensors are views into one shared underlying buffer -- mirroring how an FSDP
    DTensor's un-sharded local tensor can still alias storage with something else
    materialised by the checkpoint writer. Merging them must not leak that aliasing
    into the merged state_dict: each merged key must own independent storage.
    """
    merger = object.__new__(fsdp_merger_module.FSDPModelMerger)
    replicate = _FakeReplicate()

    shared_buffer = torch.arange(8, dtype=torch.bfloat16)
    local_a = shared_buffer[0:4]
    local_b = shared_buffer[4:8]
    assert local_a.storage().data_ptr() == local_b.storage().data_ptr()  # sanity: same storage

    merged_a = merger._merge_by_placement([local_a, local_a.clone()], replicate)
    merged_b = merger._merge_by_placement([local_b, local_b.clone()], replicate)

    # values must still be correct
    torch.testing.assert_close(merged_a.float(), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(merged_b.float(), torch.tensor([4.0, 5.0, 6.0, 7.0]))

    # ... but the two different keys' tensors must not share storage, or
    # save_pretrained will silently drop one of them as a "duplicate".
    assert merged_a.storage().data_ptr() != merged_b.storage().data_ptr()


def test_load_and_merge_state_dicts_end_to_end_no_aliasing(fsdp_merger_module, monkeypatch):
    """End-to-end regression test through the real merge entrypoint.

    Fakes two rank checkpoint files: two "replicated" (un-sharded) parameters that
    alias a shared buffer on rank 0 (mimicking the reported trigger condition), plus
    one normally FSDP-sharded parameter for realism. Asserts the merged state_dict
    has correct values and no unexpected storage aliasing between distinct keys.
    """
    module = fsdp_merger_module
    merger = object.__new__(module.FSDPModelMerger)
    merger.config = types.SimpleNamespace(local_dir="/fake/checkpoint/dir")

    replicate = _FakeReplicate()
    shard0 = _FakeShard(0)

    def fake_torch_load(path, map_location=None, weights_only=None):
        path_str = str(path)
        rank = 0 if "rank_0" in path_str else 1

        # Two distinct replicated params sharing one buffer -- only present with its
        # "true" values on rank 0 for this test; _merge_by_placement always takes
        # tensors[0] (rank 0) for replicated placements.
        shared_buffer = torch.arange(8, dtype=torch.bfloat16) + (100 if rank else 0)
        state = {
            "model.small_a.weight": _FakeDTensor(shared_buffer[0:4], (replicate,)),
            "model.small_b.weight": _FakeDTensor(shared_buffer[4:8], (replicate,)),
            "model.big.weight": _FakeDTensor(torch.full((2,), float(rank), dtype=torch.bfloat16), (shard0,)),
        }
        return state

    monkeypatch.setattr(torch, "load", fake_torch_load)

    merged = merger._load_and_merge_state_dicts(world_size=2, total_shards=2, mesh_shape=(1,), mesh_dim_names=("fsdp",))

    torch.testing.assert_close(merged["model.small_a.weight"].float(), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(merged["model.small_b.weight"].float(), torch.tensor([4.0, 5.0, 6.0, 7.0]))
    torch.testing.assert_close(merged["model.big.weight"].float(), torch.tensor([0.0, 0.0, 1.0, 1.0]))

    keys = list(merged.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = merged[keys[i]], merged[keys[j]]
            assert a.storage().data_ptr() != b.storage().data_ptr(), (
                f"keys {keys[i]!r} and {keys[j]!r} unexpectedly share tensor storage; "
                "save_pretrained would silently drop one of them as a duplicate "
                "(see https://github.com/verl-project/verl/issues/6259)"
            )
