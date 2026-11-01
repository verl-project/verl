"""Loss aggregation for the outer-chunked step.

Pure arithmetic, no torch and no megatron imports, so the invariant can be tested without
a training stack - the same reason ``chunk_window`` exists.

Every pass computes its loss against the SAME global ``loss_mask`` and zeroes the rows it did
not compute, so a pass that owns none of the supervised tokens contributes exactly 0. The
passes are therefore disjoint contributions to one mean and must be **summed**. Averaging
them divided the K=2 loss by the pass count (4.18 -> 1.39 measured at 512K) and made a
correct step look broken.
"""

from __future__ import annotations

# A per-token-scaled loss (the scan pass reports ``mean * routed_num_tokens / dp_size``)
# is not a mean and would dominate any sum; 100 is far above a real SFT mean and far below
# that scaled value.
MAX_REPORTABLE_LOSS = 100.0


def aggregate_pass_losses(pass_losses, fallback: float) -> float:
    """Sum the reportable per-pass losses, or return ``fallback`` if none qualifies.

    Args:
        pass_losses: the per-pass ``dict["loss"]`` values, in pass order.
        fallback: what to report when every entry is filtered out (``NaN`` or a scaled
            value); the caller's own total is the honest answer in that case.
    """
    values = [v for v in pass_losses if v == v and abs(v) < MAX_REPORTABLE_LOSS]
    return sum(values) if values else fallback
