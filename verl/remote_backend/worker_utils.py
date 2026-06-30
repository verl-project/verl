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
"""Backend-agnostic tensor + metric helpers shared across per-backend
forwarder workers under :mod:`verl.remote_backend.workers`.

Lifted here (out of any per-backend module) so each backend's worker
can reuse them without re-implementing nested-jagged reconstruction or
metric normalization, and without importing one another's Worker
classes or single-controller dispatch decorators.

Each per-backend worker keeps its own payload-encoding decisions on
its adapter's compute/update methods (the ABC intentionally doesn't
fix those signatures); these helpers cover the small set of generic
transformations a worker needs around those calls:

* :func:`make_njt` — reconstruct a nested-jagged tensor from a dense
  ``[B, L]`` tensor returned by a backend, using the offsets carried in
  ``data["input_ids"]``.
* :func:`normalize_backend_metrics` — coerce whatever shape a backend
  put in its ``metrics`` dict (a :class:`Metric`, a list of them, a list
  of bare scalars, or a bare scalar) into the canonical verl
  :class:`Metric` form (or a pass-through list when the trainer needs
  to aggregate later).
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor

from verl.utils.metric import AggregationType, Metric


def make_njt(data: TensorDict, tensor: Tensor) -> Tensor:
    """Reconstruct a nested-jagged tensor from a dense ``[B, L]`` tensor
    and the offsets carried in ``data["input_ids"]``."""
    cu_seqlens = data["input_ids"].offsets()
    seq_lengths = cu_seqlens.diff()
    starts = data["attention_mask"].long().argmax(dim=1)
    pieces = [tensor[b, starts[b].item() : starts[b].item() + seq_lengths[b].item()] for b in range(tensor.shape[0])]
    flat = torch.cat(pieces, dim=0)
    return torch.nested.nested_tensor_from_jagged(flat, cu_seqlens)


def shift_nested_response_aligned_to_predict_next(njt: Tensor) -> Tensor:
    """Convert a response-aligned nested tensor back to the legacy
    "predict-next" convention by left-shifting each sequence by one.

    The Arctic zorro fast path emits model output already aligned with the
    response token at each slot (slot i holds the log prob of token[i]). Shift it
    left so slot i holds log P(token[i+1]); ``no_padding_2_padding`` can then apply
    a single uniform ``shift=-1`` slice for both the zorro and standard paths,
    instead of branching the slice on a per-call ``shift`` argument."""
    v = njt.values()
    v_new = torch.empty_like(v)
    v_new[:-1] = v[1:]
    v_new[-1] = 0  # trailing slot of last seq; never read by the shift=-1 slice
    return torch.nested.nested_tensor_from_jagged(v_new, njt.offsets())


def normalize_backend_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalise backend metrics into verl's :class:`Metric` type.

    Shapes handled: a `Metric` (pass through), a `list[Metric]` (call
    `Metric.aggregate_dp`), a `list[scalar]` (pass through; trainer
    aggregates), or a bare scalar (wrap as MEAN).
    """
    out: dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, Metric):
            out[key] = val
        elif isinstance(val, list):
            if val and isinstance(val[0], Metric):
                out[key] = Metric.aggregate_dp(val)
            else:
                out[key] = val
        else:
            out[key] = Metric(value=val, aggregation=AggregationType.MEAN)
    return out
