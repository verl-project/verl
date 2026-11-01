"""Row-window arithmetic for the outer-sequence chunked step.

This module is deliberately free of torch and of megatron imports. ``transformer_impl``
pulls in the package ``__init__`` chain (torch + megatron-core), so arithmetic that lives
there cannot be exercised without a full training stack; the arithmetic below is exactly
the part of the chunk path that decides *which rows of ``log_probs`` a pass is allowed to
contribute to the loss*, and it is testable on its own.

Two modes matter:

``keep``
    Retired. ``width = cp_size * (numel // cp_size)``, ``start = index * width``: with
    ``numel`` being the whole sequence this width *is* the whole tensor, so pass 0 keeps
    every row and each later pass gets a nearly empty window. For K>1 that zeroes rows the
    pass actually computed, which corrupts both the loss and the gradient. Kept only so an
    explicit ``DSV41_OWN_WINDOW=keep`` can still reproduce the old behaviour for A/B work.

``k``
    The default, and the window the virtual-CP ownership rule implies: one K-th of the
    timeline per pass. It is the identity at K=1 (``width = numel``), so the single-pass
    path is unchanged.
    One K-th of the timeline per pass: ``width = numel // chunk_count``, ``start = index *
    width``. This is the window the virtual-CP ownership rule implies (virtual rank
    ``g * cp + r`` owns one contiguous block, so a whole pass owns ``cp`` consecutive
    blocks).

``off``
    No windowing; the caller keeps the tensor as produced.
"""

from __future__ import annotations

KEEP = "keep"
CHUNK = "k"
OFF = "off"
WINDOW_MODES = (KEEP, CHUNK, OFF)


def own_window_rows(
    numel: int,
    cp_size: int,
    chunk_count: int,
    index: int,
    mode: str = CHUNK,
) -> tuple[int, int]:
    """Half-open row range ``[start, end)`` pass ``index`` owns inside a full-sequence tensor.

    Args:
        numel: rows of the flattened ``log_probs`` tensor (the whole sequence, not one pass).
        cp_size: physical context-parallel size.
        chunk_count: number of passes (K).
        index: pass index, ``0 <= index < chunk_count``.
        mode: one of :data:`WINDOW_MODES`.
    """
    if mode == OFF:
        return 0, numel

    if mode == CHUNK:
        if chunk_count < 1:
            raise ValueError(f"chunk_count must be >= 1, got {chunk_count}")
        width = numel // chunk_count
        start = min(index * width, numel)
        # The last pass absorbs the `numel % chunk_count` remainder, otherwise the final rows
        # (which is where an SFT batch's supervised tail lives) belong to no pass at all.
        end = numel if index == chunk_count - 1 else min(start + width, numel)
        return start, end

    if mode == KEEP:
        if cp_size < 1:
            raise ValueError(f"cp_size must be >= 1, got {cp_size}")
        # `cp_size * (numel // cp_size)` is the shipped expression; it equals `numel` minus
        # the padded remainder, which is why it must not be simplified to `numel`.
        width = cp_size * (numel // cp_size)
        start = min(index * width, numel)
        return start, min(start + width, numel)

    raise ValueError(f"unknown window mode {mode!r}; expected one of {WINDOW_MODES}")
