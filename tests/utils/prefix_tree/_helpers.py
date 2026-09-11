# Copyright 2025 Meituan Ltd. and/or its affiliates
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

"""Shared helpers for the prefix-tree test suite.

- Stub installer for megatron / magi_attention / apex / transformer_engine
  (verl.utils.prefix_tree.forward hard-imports magi_attention): installed at
  import time so that BOTH conftest and mp.spawn'd worker processes (which do
  not run conftest) share one implementation.
- make_pt_batch / build_layout: the PrefixTreeMagiBatch wrapping mirrored from
  _finalize_prefix_tree_batch, shared by the magi / trie / junction / restore
  test modules.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import socket
import sys
import types


class _StubModule(types.ModuleType):
    """Stub module: attribute access returns a child stub (also module-like)."""

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        child_name = f"{self.__name__}.{item}"
        child = _make_stub(child_name)
        setattr(self, item, child)
        sys.modules[child_name] = child
        return child


def _make_stub(name: str) -> _StubModule:
    mod = _StubModule(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.__file__ = f"<stub:{name}>"
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    return mod


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Meta-path finder that auto-stubs any submodule of configured top-level packages."""

    _prefixes: tuple[str, ...] = ()

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        for prefix in cls._prefixes:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return importlib.util.spec_from_loader(fullname, loader=cls)
        return None

    @classmethod
    def create_module(cls, spec):
        return _make_stub(spec.name)

    @classmethod
    def exec_module(cls, module):
        pass  # stubs have no body


STUB_PACKAGES = ["megatron", "magi_attention", "apex", "transformer_engine"]


def install_stubs(packages=STUB_PACKAGES) -> None:
    for pkg in packages:
        try:
            importlib.util.find_spec(pkg)
        except (ModuleNotFoundError, ImportError):
            sys.modules[pkg] = _make_stub(pkg)
    _StubFinder._prefixes = tuple(packages)
    sys.meta_path = [f for f in sys.meta_path if not isinstance(f, _StubFinder)]
    sys.meta_path.insert(0, _StubFinder)


install_stubs()

import torch  # noqa: E402

from verl.utils.prefix_tree.dynamic import build_tree_dynamic, greedy_build_tries  # noqa: E402
from verl.utils.prefix_tree.magi import PackRestorationParam, PrefixTreeMagiBatch  # noqa: E402
from verl.utils.prefix_tree.utils import build_layout_from_tree_node  # noqa: E402


def make_pt_batch(params, subtrie, flex_key=None) -> PrefixTreeMagiBatch:
    """Mirror _finalize_prefix_tree_batch's PrefixTreeMagiBatch wrapping."""
    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=None,
        flex_key=flex_key,
        restoration=PackRestorationParam(
            segment_ranges=params.leaf_ranges,
            prefix_range=params.prefix_range,
            ancestor_segment_ranges=getattr(params, "_leaf_ancestor_ranges", None),
            boundary_registry=getattr(params, "boundary_registry", None),
        ),
        subtrie=subtrie,
        real_tokens=params.tree_packed_tokens.shape[0],
    )


def build_layout(samples):
    """build_tree_dynamic + build_layout_from_tree_node, wrapped into a pb. Returns (pb, params)."""
    subtrie = build_tree_dynamic(samples)
    assert subtrie is not None, "samples share a prefix, a subtrie must exist"
    params = build_layout_from_tree_node(samples, subtrie)
    return make_pt_batch(params, subtrie), params


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_grpo_samples(n_prompts, rollout_n, prefix_len, resp_len, seed=0, vocab=100_000, duplicate_pair=None):
    """GRPO-style batch: prompt shared per group, random responses.
    duplicate_pair=(p, r0, r1) makes group p's responses r0 and r1 identical."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    for p in range(n_prompts):
        prefix = torch.randint(0, vocab, (prefix_len,), generator=g)
        resps = [torch.randint(0, vocab, (resp_len,), generator=g) for _ in range(rollout_n)]
        if duplicate_pair is not None and duplicate_pair[0] == p:
            resps[duplicate_pair[1]] = resps[duplicate_pair[2]].clone()
        for resp in resps:
            samples.append(torch.cat([prefix, resp]))
    return samples


def build_trie(samples):
    seq_lists = [s.tolist() if hasattr(s, "tolist") else list(s) for s in samples]
    trie, _ = greedy_build_tries(seq_lists)
    return trie
