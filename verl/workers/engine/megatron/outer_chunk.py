"""Drive an outer-sequence chunked step from inside verl's forward step.

The exactness requirement is that every pass's backward sees every later pass's
contribution to the compressed KV it published, so the passes cannot simply be run in
forward order with an immediate backward (that is truncated BPTT). This module owns the
whole scan -> last-pass -> reverse-replay loop and lets the caller supply the forward and
the loss, so the loss definition stays in one place.

It is invoked from the K>1 branch of the engine's forward step only; an unchunked run never
reaches this file.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def row_plan(global_len: int, cp_size: int, chunk_count: int, rank: int, index: int):
    """(start, length) of the global rows pass `index` owns on physical `rank`."""
    width = cp_size * chunk_count
    if global_len % width:
        raise ValueError(f"{global_len} rows do not tile {width} virtual ranks evenly")
    per = global_len // width
    return (index * cp_size + rank) * per, per


def slice_nested(nested: Tensor, start: int, length: int) -> Tensor:
    """Rows [start, start+length) of a jagged THD tensor, splitting sequences as needed."""
    offsets = nested.offsets().tolist()
    base = offsets[0]
    pieces = []
    lens = []
    for i in range(len(offsets) - 1):
        lo, hi = max(offsets[i], start), min(offsets[i + 1], start + length)
        if hi <= lo:
            continue
        pieces.append(nested.values()[lo - base : hi - base])
        lens.append(hi - lo)
    if not pieces:
        raise ValueError(f"window [{start}, {start + length}) covers no rows")
    values = torch.cat(pieces, dim=0)
    if values.shape[0] != length:
        raise ValueError(f"window covered {values.shape[0]} rows, expected {length}")
    return torch.nested.nested_tensor_from_jagged(values, lengths=torch.tensor(lens))


def run_chunked_step(
    *,
    run_pass: Callable[[int], Tensor],
    tape,
    chunk_count: int,
    backward: Callable[[Tensor], None],
    profile_phase: Callable[[str], None] | None = None,
) -> torch.Tensor:
    """scan -> last pass (grad) -> reverse replay, one pass's graph alive at a time.

    ``run_pass(index)`` must run that pass's forward and return its scalar loss.
    ``backward(loss)`` runs the backward for one loss, accumulated into the shared
    parameter gradients exactly like micro-batching does.
    The no-grad prefill loss is ignored; only grad passes contribute to the total.
    Callers collecting loss metrics must likewise exclude prefill invocations.
    """
    if chunk_count < 2:
        raise ValueError("chunk_count < 2 must use the ordinary path")

    for index in range(chunk_count - 1):
        tape.begin(index, "prefill")
        with torch.no_grad():
            run_pass(index)
        if profile_phase is not None:
            profile_phase(f"chunk_{index}_forward_end")

    tape.begin(chunk_count - 1, "grad")
    with torch.enable_grad():
        last = run_pass(chunk_count - 1)
    if profile_phase is not None:
        profile_phase(f"chunk_{chunk_count - 1}_forward_end")
        profile_phase(f"chunk_{chunk_count - 1}_backward_start")
    backward(last)
    if profile_phase is not None:
        profile_phase(f"chunk_{chunk_count - 1}_backward_end")
    tape.collect_all_adjoints()
    total = last.detach()

    for index in reversed(range(chunk_count - 1)):
        tape.begin(index, "grad")
        with torch.enable_grad():
            loss = run_pass(index)
        if profile_phase is not None:
            profile_phase(f"chunk_{index}_forward_end")
            profile_phase(f"chunk_{index}_backward_start")
        backward(loss)
        if profile_phase is not None:
            profile_phase(f"chunk_{index}_backward_end")
        tape.collect_all_adjoints()
        total = total + loss.detach()
    return total
