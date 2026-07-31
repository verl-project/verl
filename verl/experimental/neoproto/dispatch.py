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
"""NeoProto-aware dispatch helpers for v0 single-controller workers.

Before Ray pickles a chunked NeoProto, attach per-rank ``OBJ_REF`` /
``LOCAL_REF`` ObjectRefs so ``NeoProto.__getstate__`` ships masked ref tables
instead of the full column. This is a connection between ``NeoProto`` and
``StorageEngine`` for cheap Ray pickle.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np
import ray

from verl.experimental.neoproto.neo import NeoProto
from verl.experimental.neoproto.storage.engine import Ref, RefTable

_NEO_DISPATCH_ENABLED = False


def enable_neo_dispatch(enabled: bool = True) -> None:
    """Toggle NeoProto ref-table preserialize on dataproto dispatch paths."""
    global _NEO_DISPATCH_ENABLED
    _NEO_DISPATCH_ENABLED = bool(enabled)


def is_neo_dispatch_enabled() -> bool:
    return _NEO_DISPATCH_ENABLED


def is_neo_batch(obj: Any) -> bool:
    return getattr(obj, "__neoproto__", False) or isinstance(obj, NeoProto)


def attach_preserialized_ref_tables(
    neoproto: NeoProto,
    neo_chunk_list: Sequence[NeoProto],
    *,
    sp_size: int = 1,
) -> None:
    """Attach ``OBJ_REF`` / ``LOCAL_REF`` onto each chunk for cheap Ray pickle."""
    chunks = len(neo_chunk_list)
    assert chunks % sp_size == 0, f"chunks={chunks} not divisible by sp_size={sp_size}"
    dp_size = chunks // sp_size

    rt = neoproto.ref_table
    bs = rt.batch_size
    order = neoproto.dim0_index.sample_indices
    if order is None:
        order = np.arange(bs, dtype=np.int64)
    total = len(order)
    assert total % dp_size == 0, f"n_samples={total} not divisible by dp_size={dp_size}"
    per = total // dp_size

    objref_keys = []
    for k in rt.keys():
        col = rt[k]
        if (
            isinstance(col, np.ndarray)
            and len(col) > 0
            and isinstance(col[0], Ref)
            and isinstance(col[0].dataptr, ray.ObjectRef)
        ):
            objref_keys.append(k)

    dp_positions = [order[g * per : (g + 1) * per] for g in range(dp_size)]

    local_refs = {}
    for k in rt.keys():
        if k not in objref_keys:
            local_refs[k] = rt[k]
    local_ref_table = ray.put(RefTable(local_refs, batch_size=bs))

    def _build_and_put(rank: int):
        positions = dp_positions[rank // sp_size]
        new_refs = {}
        for k in rt.keys():
            col = rt[k]
            if k in objref_keys:
                masked = np.empty(len(col), dtype=object)
                masked[positions] = col[positions]
                new_refs[k] = masked
        return ray.put(RefTable(new_refs, batch_size=bs))

    n_workers = min(chunks, 32)
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
        rank_ref_tables = list(ex.map(_build_and_put, range(chunks)))

    assert len(rank_ref_tables) == chunks
    for i, ref_table in enumerate(rank_ref_tables):
        neo_chunk_list[i].OBJ_REF = ref_table
        neo_chunk_list[i].LOCAL_REF = local_ref_table


def maybe_attach_neo_ref_tables(original: Any, chunks: Sequence[Any], *, sp_size: int = 1) -> None:
    """No-op unless neo dispatch is enabled and ``original`` is a NeoProto batch."""
    # Avoid Ray's implicit auto-init in local/offline paths. Production worker
    # dispatch already runs under an initialized Ray driver, while unit tests
    # and in-memory debugging should remain usable without starting a cluster.
    if not _NEO_DISPATCH_ENABLED or not is_neo_batch(original) or not ray.is_initialized():
        return
    attach_preserialized_ref_tables(original, chunks, sp_size=sp_size)
