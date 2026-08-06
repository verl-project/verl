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
"""Worker-side NeoProto ↔ TensorDict bridge for v0 engine workers.

Transfer plane: Ray RPC carries NeoProto (refs). Compute plane: engines still
consume TensorDict. This module materializes inside the worker process only.

When ``required_keys`` is provided, only those columns (plus present
``optional_keys``) are fetched from storage — column-level lazy load without
changing engine ``_impl`` code. ``required_keys=None`` keeps the legacy full
``to_tensordict()`` behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from verl.workers.utils.batch_adapter import EngineBatchContext, EngineBatchSpec
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

_FORCE_FULL_MATERIALIZE = os.environ.get("NEO_BRIDGE_FULL_MATERIALIZE", "0") == "1"
_FP32_ENGINE_OUTPUT_KEYS = frozenset({"values", "log_probs", "entropy", "sum_pi_squared"})


@dataclass
class _NeoEngineCtx:
    padded_td: TensorDict
    restore_keys: tuple[str, ...]


def _neo_has_key(data: Any, key: str) -> bool:
    ref_table = getattr(data, "ref_table", None)
    if ref_table is None:
        return False
    return key in ref_table


def _is_object_non_tensor_key(data: Any, key: str) -> bool:
    schema = getattr(data, "schema", None) or {}
    spec = schema.get(key)
    if spec is None:
        return False
    return getattr(spec, "dtype", None) in ("object", "bytes")


def _neo_to_tensordict_subset(data: Any, keys: Sequence[str]) -> TensorDict:
    """Materialize only ``keys`` into a TensorDict (same packing as ``to_tensordict``)."""
    from tensordict.tensorclass import NonTensorData, NonTensorStack

    from verl.experimental.neoproto.neo import FULL_GRANULARITY
    from verl.utils import tensordict_utils as tu

    meta: dict[str, Any] = {}
    batch_keys: list[str] = []
    for key in keys:
        if not _neo_has_key(data, key):
            continue
        spec = (getattr(data, "schema", None) or {}).get(key)
        if spec is not None and getattr(spec, "granularity", None) == FULL_GRANULARITY:
            meta[key] = data.meta_info[key]
        else:
            batch_keys.append(key)

    tensors: dict[str, Any] = {}
    if batch_keys:
        # One materialize pass for all non-meta columns.
        mats = data.materialize(batch_keys)
        for key, val in mats.items():
            if _is_object_non_tensor_key(data, key):
                if isinstance(val, list):
                    arr = np.empty(len(val), dtype=object)
                    for i, item in enumerate(val):
                        arr[i] = item
                    val = arr
                assert isinstance(val, np.ndarray)
                tensors[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])
            else:
                tensors[key] = val

    return tu.get_tensordict(tensor_dict=tensors, non_tensor_dict=meta)


def prepare_engine_input(
    data: Any,
    *,
    restore_padding_keys: Optional[Iterable[str]] = None,
    required_keys: Optional[Iterable[str]] = None,
    optional_keys: Optional[Iterable[str]] = None,
) -> tuple[TensorDict, _NeoEngineCtx]:
    """Convert inbound RPC data to an engine TensorDict.

    NeoProto batches are materialized and converted with
    ``left_right_2_no_padding`` inside the worker. The generic worker adapter
    passes classic TensorDict inputs directly to the engine without calling
    this function.

    Args:
        required_keys: If ``None``, full ``to_tensordict()`` (legacy). Otherwise
            these keys must exist on the Neo batch and will be materialized.
        optional_keys: Extra keys materialized only when present on the batch.
    """
    if required_keys is None or _FORCE_FULL_MATERIALIZE:
        padded_td = data.to_tensordict()
    else:
        req = tuple(required_keys)
        opt = tuple(optional_keys or ())
        missing = [k for k in req if not _neo_has_key(data, k)]
        if missing:
            raise KeyError(
                f"NeoProto worker bridge missing required keys for engine input: {missing}. "
                f"present={sorted(getattr(data, 'ref_table', {}).keys())}"
            )
        keys = list(req)
        for k in opt:
            if k not in keys and _neo_has_key(data, k):
                keys.append(k)
        padded_td = _neo_to_tensordict_subset(data, keys)

    engine_td = left_right_2_no_padding(padded_td)
    keys = tuple(restore_padding_keys or ())
    return engine_td, _NeoEngineCtx(padded_td=padded_td, restore_keys=keys)


def finalize_engine_output(
    output: Any,
    ctx: _NeoEngineCtx,
) -> Any:
    """Restore padding on tensor outputs and wrap as NeoProto when input was Neo."""
    if output is None:
        return output

    if torch.is_tensor(output):
        # Unexpected; return as-is
        return output

    # Support TensorDict-like outputs from TrainingWorker.
    if isinstance(output, TensorDict) or hasattr(output, "keys"):
        for key in _FP32_ENGINE_OUTPUT_KEYS:
            if key in output.keys() and torch.is_tensor(output[key]):
                output[key] = output[key].float()
        for key in ctx.restore_keys:
            if key in output.keys():
                val = output[key]
                if torch.is_tensor(val):
                    output[key] = no_padding_2_padding(val, ctx.padded_td)
        from verl.experimental.neoproto import DataProto as NeoDataProto

        if isinstance(output, TensorDict):
            return NeoDataProto.from_tensordict(output)
        # Already a batch container
        if isinstance(output, NeoDataProto):
            return output
        return NeoDataProto.from_tensordict(TensorDict(dict(output), batch_size=output.batch_size))

    return output


def prepare_neo_engine_batch(data: Any, spec: EngineBatchSpec) -> EngineBatchContext:
    """Build the generic worker adapter context for a NeoProto batch."""
    engine_td, neo_ctx = prepare_engine_input(
        data,
        restore_padding_keys=spec.restore_padding_keys,
        required_keys=spec.required_keys,
        optional_keys=spec.optional_keys,
    )
    return EngineBatchContext(
        payload=engine_td,
        finalize=lambda output: finalize_engine_output(output, neo_ctx),
    )
