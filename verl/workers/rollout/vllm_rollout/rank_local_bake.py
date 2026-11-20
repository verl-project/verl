# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Which part of each HF tensor this vLLM TP rank keeps, for ``rank_local_named_tensors``.

With it the ``sharded_nixl`` trainer sends a TP rank only its own box of a tensor. Instead of
per-model rules, a dry run of ``model.load_weights`` feeds each checkpoint name as a storage-less
fake tensor that only records ``narrow`` calls, and catches the loader's final ``copy_``. A name
whose loader copied exactly one strict sub-box into exactly one parameter view gets a ``slices``
entry (advertised in ``RolloutLayout.slices``) and a :class:`Scatter` (where a received box lands).
Every other name (copied whole, several copies, more than 2 dims, any other op on the weight,
loader error) is sent full and goes through ``load_weights``.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True)
class Scatter:
    """Where a received box goes: ``param.data.as_strided(shape, stride, param.storage_offset() + offset)``."""

    param: str
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass
class RankLocalPlan:
    slices: dict[str, dict] = field(default_factory=dict)
    scatters: dict[str, Scatter] = field(default_factory=dict)


class _Unsupported(RuntimeError):
    pass


class _BoxFake(torch.Tensor):
    """No storage; carries the box ``[(start, length), ...]`` of the full HF tensor it stands for."""

    @staticmethod
    def __new__(cls, box, device, sink):
        t = torch.Tensor._make_wrapper_subclass(cls, [n for _, n in box], dtype=torch.bfloat16, device=device)
        t._box, t._sink = list(box), sink
        return t

    def _narrow(self, dim, start, length):
        dim %= self.dim()
        s, n = self._box[dim]
        if start < 0 or start + length > n:
            raise _Unsupported(f"narrow({dim}, {start}, {length}) out of {n}")
        box = list(self._box)
        box[dim] = (s + start, length)
        return _BoxFake(box, self.device, self._sink)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.copy_:
            src = args[1] if len(args) > 1 else kwargs["src"]
            if isinstance(args[0], cls) or not isinstance(src, cls):
                raise _Unsupported("copy_ into a fake")
            src._sink((src._box, args[0]))
            return args[0]
        if func in (torch.Tensor.narrow, torch.narrow) and isinstance(args[0], cls):
            return args[0]._narrow(*args[1:], **kwargs)
        # metadata (shape, dim, numel, ...) works on the wrapper; any op that needs data reaches dispatch
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise _Unsupported(f"{func} on the weight")


def read_manifest(model_path: str) -> dict[str, list[int]]:
    """HF name -> full shape, from the safetensors headers only (no tensor data is read)."""
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        logger.warning("rank-local bake: no *.safetensors in %r, every tensor is sent full", model_path)
    manifest = {}
    for f in files:
        with safe_open(f, framework="pt") as st:
            for k in st.keys():
                manifest[k] = list(st.get_slice(k).get_shape())
    return manifest


def bake(model: torch.nn.Module, model_path: str) -> RankLocalPlan:
    """Dry-run ``model.load_weights`` once per checkpoint name; see the module docstring.

    Runs on the real parameters: a loader may still write before it reaches the weight (e.g. zeroing
    the vocab padding rows), which is harmless because the trainer sends every weight right after.
    """
    device = next(model.parameters()).device
    owners = {p.untyped_storage().data_ptr(): (n, p) for n, p in model.named_parameters()}
    plan, errors = RankLocalPlan(), {}
    for name, shape in read_manifest(model_path).items():
        copies = []
        try:
            with torch.no_grad():
                model.load_weights([(name, _BoxFake([(0, n) for n in shape], device, copies.append))])
        except Exception as e:  # any loader we cannot follow keeps the full tensor
            errors[name] = f"{type(e).__name__}: {e}"
            continue
        if len(copies) != 1:  # 0: skipped by the model (e.g. tied lm_head); >1: several destinations
            continue
        box, dest = copies[0]
        lens = tuple(n for _, n in box)
        owner = owners.get(dest.untyped_storage().data_ptr())
        # the trainer's box planner cuts at most 2-D tensors (e.g. not stacked 3-D experts)
        if owner is None or tuple(dest.shape) != lens or list(lens) == shape or len(shape) > 2:
            continue
        pname, p = owner
        plan.slices[name] = {"full_shape": shape, "box": [list(b) for b in box]}
        plan.scatters[name] = Scatter(pname, dest.storage_offset() - p.storage_offset(), lens, tuple(dest.stride()))
    logger.info(
        "rank-local bake: %d names sliced, %d unsupported%s",
        len(plan.slices),
        len(errors),
        f" (e.g. {next(iter(errors.items()))})" if errors else "",
    )
    return plan


def apply(model: torch.nn.Module, plan: RankLocalPlan, weights):
    """Copy every received box straight into its parameter view; return the rest for ``load_weights``.

    Dispatches on shape: a tensor with its box's local shape is rank-local, a full-shape one (a
    ``named_tensors`` sync, or a name without a box) goes to the loader.
    """
    params, rest = dict(model.named_parameters()), []
    for name, t in weights:
        sc = plan.scatters.get(name)
        if sc is None or tuple(t.shape) != sc.shape:
            rest.append((name, t))
            continue
        p = params[sc.param]
        p.data.as_strided(sc.shape, sc.stride, p.storage_offset() + sc.offset).copy_(t)
    return rest
