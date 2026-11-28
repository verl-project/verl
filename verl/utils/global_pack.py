# Copyright (c) 2026 Google LLC. All rights reserved.
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
"""Global bin packing of a batch across data-parallel ranks (``VERL_TPU_GLOBAL_PACK=1``).

Static-shape engines (the TPU TorchTitan engine with ``VERL_TPU_BIN_SLOT_LEN``) run every micro-batch as one
``bin_len``-token row and pack each rank's sequences into such rows with first-fit decreasing (FFD). Each rank then
runs as many micro-batches as the busiest rank needs, so the step costs ``max_r FFD(rank r)`` micro-batches, and with
per-rank packing over the ~8 sequences of a contiguous dispatch chunk that is often well above the global bound
``ceil(total_tokens / (dp * bin_len))``.

``get_global_pack_partitions`` chooses WHICH sequences each rank gets (exactly ``len / dp`` each, as verl's equal-size
dispatch requires) so that the per-rank FFD needs as few bins as possible on the worst rank. It only returns a
permutation, like ``get_seqlen_balanced_partitions``: the dispatch, the per-rank mini-batch iterator and the engine's
own packing are unchanged. The engine packs each rank with the same ``ffd_pack_bins`` (same lengths, alignment and
segment cap), so the bin count planned here is the micro-batch count the engine runs.

Stdlib only, so the single controller can import it cheaply.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

GLOBAL_PACK_ENV = "VERL_TPU_GLOBAL_PACK"
GLOBAL_PACK_BIN_LEN_ENV = "VERL_TPU_GLOBAL_PACK_BIN_LEN"
GLOBAL_PACK_ALIGN_ENV = "VERL_TPU_GLOBAL_PACK_ALIGN"
GLOBAL_PACK_MAX_SEGMENTS_ENV = "VERL_TPU_GLOBAL_PACK_MAX_SEGMENTS"
BIN_SLOT_LEN_ENV = "VERL_TPU_BIN_SLOT_LEN"
# The GatedDeltaNet (Tokamax KDA) chunk size: every packed document starts on a 64-token chunk boundary inside the
# kernel, so budgeting each segment at its 64-aligned length keeps a plan valid for an aligned row layout too.
DEFAULT_ALIGN = 64


def global_pack_enabled() -> bool:
    return os.environ.get(GLOBAL_PACK_ENV, "0").strip().lower() in ("1", "true", "yes")


def global_pack_settings() -> tuple[int, int, int]:
    """``(bin_len, align, max_segments)`` from the environment; ``bin_len == 0`` when no bin length is configured."""
    bin_len = int(os.environ.get(GLOBAL_PACK_BIN_LEN_ENV, "") or os.environ.get(BIN_SLOT_LEN_ENV, "") or 0)
    align = int(os.environ.get(GLOBAL_PACK_ALIGN_ENV, "") or DEFAULT_ALIGN)
    max_segments = int(os.environ.get(GLOBAL_PACK_MAX_SEGMENTS_ENV, "") or 0)
    return bin_len, max(1, align), max(0, max_segments)


def segment_slot_len(length: int, bin_len: int, align: int) -> int:
    """Tokens a sequence occupies in a bin: its length rounded up to ``align`` (capped at ``bin_len``)."""
    length = int(length)
    if length > bin_len:
        raise ValueError(f"sequence of {length} tokens does not fit a {bin_len}-token bin")
    return min(bin_len, -(-length // align) * align)


def ffd_pack_bins(lengths: Sequence[int], bin_len: int, align: int = 1, max_segments: int = 0) -> list[list[int]]:
    """First-fit decreasing of ``lengths`` into ``bin_len``-token bins; returns the item indices of each bin.

    Items are sorted by decreasing slot length (stable, so ties keep index order) and each goes into the first bin with
    room (and fewer than ``max_segments`` items when ``max_segments > 0``). With ``align == 1`` and no segment cap
    this is exactly the engine's historical per-rank FFD.
    """
    slots = [segment_slot_len(ln, bin_len, align) for ln in lengths]
    bins: list[list[int]] = []
    used: list[int] = []
    for idx in sorted(range(len(slots)), key=lambda k: -slots[k]):
        ln = slots[idx]
        for b in range(len(bins)):
            if used[b] + ln <= bin_len and (max_segments <= 0 or len(bins[b]) < max_segments):
                bins[b].append(idx)
                used[b] += ln
                break
        else:
            bins.append([idx])
            used.append(ln)
    return bins


def _ffd_count(slots: Sequence[int], bin_len: int, max_segments: int) -> int:
    """Number of FFD bins for already aligned ``slots`` (the hot loop of the planner)."""
    used: list[int] = []
    cnt: list[int] = []
    for ln in sorted(slots, reverse=True):
        for b in range(len(used)):
            if used[b] + ln <= bin_len and (max_segments <= 0 or cnt[b] < max_segments):
                used[b] += ln
                cnt[b] += 1
                break
        else:
            used.append(ln)
            cnt.append(1)
    return len(used)


def bin_lower_bound(slots: Sequence[int], bin_len: int, max_segments: int = 0) -> int:
    """A lower bound on the bins any packing of ``slots`` needs (token volume, items > bin_len / 2, segment cap)."""
    if not slots:
        return 0
    bound = max(-(-sum(slots) // bin_len), sum(1 for s in slots if 2 * s > bin_len))
    if max_segments > 0:
        bound = max(bound, -(-len(slots) // max_segments))
    return bound


@dataclass
class GlobalPackStats:
    """Summary of a global packing plan (all counts in bins, i.e. static micro-batch rows)."""

    dp_size: int
    bin_len: int
    num_mini_batches: int
    real_tokens: int
    # Per mini-batch: the busiest rank's bins, which is the micro-batches every rank runs.
    max_bins_per_rank: list[int] = field(default_factory=list)
    # Per mini-batch: max(ceil(global bin lower bound / dp), the busiest rank's own lower bound).
    lower_bound_per_rank: list[int] = field(default_factory=list)
    # Per mini-batch: ceil(tokens / (dp * bin_len)), the bound when sequences could be split freely.
    token_bound_per_rank: list[int] = field(default_factory=list)

    @property
    def micro_batches(self) -> int:
        return sum(self.max_bins_per_rank)

    @property
    def fill_ratio(self) -> float:
        rows = self.micro_batches * self.dp_size
        return self.real_tokens / (rows * self.bin_len) if rows else 0.0

    def as_metrics(self, prefix: str = "global_pack") -> dict[str, float]:
        return {
            f"{prefix}/micro_batches_per_rank": float(self.micro_batches),
            f"{prefix}/lower_bound_per_rank": float(sum(self.lower_bound_per_rank)),
            f"{prefix}/token_bound_per_rank": float(sum(self.token_bound_per_rank)),
            f"{prefix}/fill_ratio": float(self.fill_ratio),
        }

    def summary(self) -> str:
        return (
            f"global pack: dp={self.dp_size} bin_len={self.bin_len} mini_batches={self.num_mini_batches} "
            f"micro_batches/rank={self.micro_batches} (lower bound {sum(self.lower_bound_per_rank)}, token bound "
            f"{sum(self.token_bound_per_rank)}) real_tokens={self.real_tokens} fill={self.fill_ratio:.3f}"
        )


class _RankPlanner:
    """Assigns ``n`` items (aligned slot lengths) to ``dp`` ranks of exactly ``n / dp`` items each.

    The cost of a rank is the FFD bin count of its items; the objective is the maximum over ranks (then the number of
    ranks at that maximum). Two initial assignments are improved by pairwise swaps that lower a worst rank without
    raising the partner to the maximum; the better final plan wins.
    """

    def __init__(self, slots: list[int], dp: int, bin_len: int, max_segments: int, max_swap_rounds: int = 0):
        self.slots = slots
        self.dp = dp
        self.k = len(slots) // dp
        self.bin_len = bin_len
        self.max_segments = max_segments
        self.max_swap_rounds = max_swap_rounds or 4 * len(slots)
        self._cache: dict[tuple[int, ...], int] = {}
        self.global_lb = -(-bin_lower_bound(slots, bin_len, max_segments) // dp)

    def cost(self, items: list[int]) -> int:
        key = tuple(sorted((self.slots[i] for i in items), reverse=True))
        val = self._cache.get(key)
        if val is None:
            val = _ffd_count(key, self.bin_len, self.max_segments)
            self._cache[key] = val
        return val

    # ---------------------------------------------------------------- initial assignments
    def snake_init(self) -> list[list[int]]:
        """Deal items by decreasing length in a boustrophedon over the ranks (equal counts, similar size mix)."""
        order = sorted(range(len(self.slots)), key=lambda i: -self.slots[i])
        ranks: list[list[int]] = [[] for _ in range(self.dp)]
        for pos, idx in enumerate(order):
            rnd, off = divmod(pos, self.dp)
            ranks[off if rnd % 2 == 0 else self.dp - 1 - off].append(idx)
        return ranks

    def bins_init(self) -> list[list[int]]:
        """Global FFD into bins, then deal whole bins to ranks honoring the per-rank item count (load balanced)."""
        bins = ffd_pack_bins(self.slots, self.bin_len, 1, self.max_segments)
        per_rank = -(-len(bins) // self.dp)
        need = [self.k] * self.dp
        nbins = [0] * self.dp
        ranks: list[list[int]] = [[] for _ in range(self.dp)]
        leftover: list[int] = []
        for b in sorted(bins, key=lambda b: (-len(b), -sum(self.slots[i] for i in b))):
            best, best_key = -1, None
            for r in range(self.dp):
                slots_left = per_rank - nbins[r]
                if slots_left <= 0 or need[r] < len(b):
                    continue
                key = (need[r] / slots_left, -nbins[r])
                if best_key is None or key > best_key:
                    best, best_key = r, key
            if best < 0:
                leftover.extend(b)
                continue
            ranks[best].extend(b)
            need[best] -= len(b)
            nbins[best] += 1
        for idx in sorted(leftover, key=lambda i: -self.slots[i]):
            cands = [r for r in range(self.dp) if need[r] > 0]
            r = min(cands, key=lambda r: (self.cost(ranks[r] + [idx]) - self.cost(ranks[r]), self.cost(ranks[r])))
            ranks[r].append(idx)
            need[r] -= 1
        return ranks

    # ---------------------------------------------------------------- local search
    def improve(self, ranks: list[list[int]]) -> list[list[int]]:
        ranks = [list(r) for r in ranks]
        costs = [self.cost(r) for r in ranks]
        sums = [sum(self.slots[i] for i in r) for r in ranks]
        L = self.bin_len
        for _ in range(self.max_swap_rounds):
            worst = max(costs)
            if worst <= self.global_lb:
                break
            moved = False
            for r in [r for r in range(self.dp) if costs[r] == worst]:
                target = worst - 1
                # Partners with the most slack first.
                partners = sorted((s for s in range(self.dp) if costs[s] < worst), key=lambda s: (costs[s], sums[s]))
                for i in sorted(ranks[r], key=lambda i: -self.slots[i]):
                    li = self.slots[i]
                    # r needs sum_r - li + lj <= target * L (volume) to drop a bin.
                    lj_max = target * L - sums[r] + li
                    for s in partners:
                        for j in ranks[s]:
                            lj = self.slots[j]
                            # Only a smaller item helps r; s must keep its volume below `worst` bins.
                            if lj >= li or lj > lj_max or sums[s] - lj + li > (worst - 1) * L:
                                continue
                            new_r = [x for x in ranks[r] if x != i] + [j]
                            if self.cost(new_r) > target:
                                continue
                            new_s = [x for x in ranks[s] if x != j] + [i]
                            cs = self.cost(new_s)
                            if cs >= worst:
                                continue
                            ranks[r], ranks[s] = new_r, new_s
                            costs[r], costs[s] = self.cost(new_r), cs
                            sums[r], sums[s] = sums[r] - li + lj, sums[s] - lj + li
                            moved = True
                            break
                        if moved:
                            break
                    if moved:
                        break
                if moved:
                    break
            if not moved:
                break
        return ranks

    def score(self, ranks: list[list[int]]) -> tuple[int, int, int]:
        costs = [self.cost(r) for r in ranks]
        worst = max(costs)
        return worst, sum(1 for c in costs if c == worst), sum(costs)

    def plan(self) -> list[list[int]]:
        best = None
        for init in (self.bins_init, self.snake_init):
            cand = self.improve(init())
            if best is None or self.score(cand) < self.score(best):
                best = cand
            if self.score(best)[0] <= self.global_lb:
                break
        return best


def get_global_pack_partitions(
    seqlen_list: Sequence[int],
    k_partitions: int,
    bin_len: int,
    align: int = DEFAULT_ALIGN,
    max_segments: int = 0,
    num_mini_batches: int = 1,
) -> tuple[list[list[int]], GlobalPackStats]:
    """Equal-size partitions of ``seqlen_list`` over ``k_partitions`` DP ranks that minimize per-rank FFD bins.

    Drop-in for ``get_seqlen_balanced_partitions(..., equal_size=True)``: reorder the batch with
    ``[j for p in partitions for j in p]`` and verl's contiguous dispatch hands partition ``r`` to DP rank ``r``.

    ``num_mini_batches > 1`` packs each contiguous mini-batch slice ``[m * n / M, (m + 1) * n / M)`` separately; the
    indices of a partition stay sorted, so rank ``r``'s ``m``-th mini-batch (a contiguous slice of its chunk when the
    actor does not shuffle) is exactly its share of mini-batch ``m``.

    Returns ``(partitions, stats)``.
    """
    n = len(seqlen_list)
    if bin_len <= 0:
        raise ValueError(f"bin_len must be positive, got {bin_len}")
    if num_mini_batches <= 0 or n % (k_partitions * num_mini_batches):
        raise ValueError(
            f"{n} sequences do not split evenly into {num_mini_batches} mini-batches x {k_partitions} ranks"
        )
    slots = [segment_slot_len(ln, bin_len, align) for ln in seqlen_list]
    per_mb = n // num_mini_batches
    partitions: list[list[int]] = [[] for _ in range(k_partitions)]
    stats = GlobalPackStats(
        dp_size=k_partitions,
        bin_len=bin_len,
        num_mini_batches=num_mini_batches,
        real_tokens=int(sum(int(x) for x in seqlen_list)),
    )
    for m in range(num_mini_batches):
        base = m * per_mb
        mb_slots = slots[base : base + per_mb]
        planner = _RankPlanner(mb_slots, k_partitions, bin_len, max_segments)
        ranks = planner.plan()
        costs = [planner.cost(r) for r in ranks]
        worst = max(costs)
        worst_lb = max(bin_lower_bound([mb_slots[i] for i in r], bin_len, max_segments) for r in ranks)
        stats.max_bins_per_rank.append(worst)
        stats.lower_bound_per_rank.append(max(planner.global_lb, worst_lb))
        mb_tokens = sum(int(x) for x in seqlen_list[base : base + per_mb])
        stats.token_bound_per_rank.append(math.ceil(mb_tokens / (k_partitions * bin_len)))
        for r, items in enumerate(ranks):
            partitions[r].extend(base + i for i in items)
    partitions = [sorted(p) for p in partitions]
    seen = sorted(i for p in partitions for i in p)
    assert seen == list(range(n)), "global pack must assign every sequence exactly once"
    assert all(len(p) == n // k_partitions for p in partitions)
    return partitions, stats


def global_pack_partitions_from_env(
    seqlen_list: Sequence[int], k_partitions: int, num_mini_batches: int = 1
) -> tuple[list[list[int]], GlobalPackStats] | None:
    """``get_global_pack_partitions`` with the environment's settings; ``None`` when no bin length is configured."""
    bin_len, align, max_segments = global_pack_settings()
    if bin_len <= 0:
        logger.warning(
            "%s=1 but neither %s nor %s is set; keeping the default balancing",
            GLOBAL_PACK_ENV,
            GLOBAL_PACK_BIN_LEN_ENV,
            BIN_SLOT_LEN_ENV,
        )
        return None
    return get_global_pack_partitions(
        seqlen_list,
        k_partitions,
        bin_len,
        align=align,
        max_segments=max_segments,
        num_mini_batches=num_mini_batches,
    )
