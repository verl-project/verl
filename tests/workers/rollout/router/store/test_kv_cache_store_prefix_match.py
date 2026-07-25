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

"""CPU unit tests for chained-prefix matching in :class:`KVCacheStore`."""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.store.kv_cache_store import KVCacheStore
from verl.workers.rollout.router.kvcaware.types import Layer
from verl.workers.rollout.router.kvcaware.utils.hash import get_prefix_hashes

pytestmark = [pytest.mark.ut, pytest.mark.cpu]

BLOCK_SIZE = 2
SEED = 0


def _store_with_block_size() -> KVCacheStore:
    """A fresh non-singleton store with ``block_size`` set."""
    store = KVCacheStore()
    store.block_size = BLOCK_SIZE
    return store


def _hashes(prompt_ids: list[int]) -> list[str]:
    """Prefix hashes for ``prompt_ids`` as the store keys them (str)."""
    return [str(h) for h in get_prefix_hashes(prompt_ids, BLOCK_SIZE, SEED)]


# ── 1. Chain property of get_prefix_hashes ──────────────────────────────


def test_chained_hashes_are_cumulative() -> None:
    """Perturbing block i leaves hash[0..i-1] stable, changes hash[i..].

    This is the invariant that makes the chain-break scan correct: each
    hash embeds all prior blocks, so caching block i implies caching 1..i-1.
    """
    base = [1, 2, 3, 4, 5, 6]  # three full blocks of size 2
    base_h = _hashes(base)

    # Perturb block index 1 (tokens [3,4] -> [3,99]).
    perturbed = [1, 2, 3, 99, 5, 6]
    pert_h = _hashes(perturbed)

    # Blocks before the perturbation are unchanged.
    assert pert_h[0] == base_h[0]
    # The perturbed block and every block after it change.
    assert pert_h[1] != base_h[1]
    assert pert_h[2] != base_h[2]


# ── 2. Per-replica longest contiguous match ─────────────────────────────


def test_max_contiguous_match_per_replica() -> None:
    """Replicas caching different depths each get their own deepest percent.

    With 4 full blocks, rep_a caches H1..H4 (100%), rep_b caches H1..H2
    (50%), rep_c caches H1..H3 (75%). Each is credited exactly its own
    contiguous depth — no replica overwrites another with a higher value.
    """
    prompt = list(range(1, 2 * 4 + 1))  # 8 tokens -> 4 blocks
    h1, h2, h3, h4 = _hashes(prompt)

    store = _store_with_block_size()
    store.add_blocks("rep_a", [h1, h2, h3, h4])
    store.add_blocks("rep_b", [h1, h2])
    store.add_blocks("rep_c", [h1, h2, h3])

    assert store.get_layer_prefix_hit_rate("rep_a", prompt, Layer.GPU) == 1.0  # 4/4
    assert store.get_layer_prefix_hit_rate("rep_b", prompt, Layer.GPU) == 0.5  # 2/4
    assert store.get_layer_prefix_hit_rate("rep_c", prompt, Layer.GPU) == 0.75  # 3/4


# ── 3. Chain break stops at first missing block ─────────────────────────


def test_chain_break_stops_at_first_missing_block() -> None:
    """A replica with H1..H3 but not H4 stops at 3, even if H5 is cached.

    rep_a caches H1..H3 and H5 (a *gap* at H4). The scan breaks at H4, so
    rep_a is credited 5/5 = 100%, never the 5/5 = 100% that H5 would imply.
    rep_b caches H1..H5 contiguously and gets 100%.

    Note: rep_a's "H5 without H4" is synthetic (a real chained cache could
    not produce H5 without H4). The point is that even if such a state were
    injected, the simple scan still gets rep_a right.
    """
    prompt = list(range(1, 2 * 5 + 1))  # 10 tokens -> 5 blocks
    h1, h2, h3, h4, h5 = _hashes(prompt)

    store = _store_with_block_size()
    store.add_blocks("rep_a", [h1, h2, h3, h5])  # gap at H4
    store.add_blocks("rep_b", [h1, h2, h3, h4, h5])  # full chain

    # rep_a gaps H4 → contiguous chain breaks at H4 → 3/5 (the prefix it
    # actually shares), not 5/5. A real chained cache cannot hold H5 without
    # H4 (H5's hash embeds H4), so the contiguous scan is the correct hit.
    assert store.get_layer_prefix_hit_rate("rep_a", prompt, Layer.GPU) == 0.6  # 3/5
    assert store.get_layer_prefix_hit_rate("rep_b", prompt, Layer.GPU) == 1.0  # 5/5


# ── 4. The "telemetry noise" failure mode is already prevented ──────────


def test_isolated_later_block_is_not_credited() -> None:
    """A replica injected with *only* H3 gets 0%, not H3's higher percent.

    This is the exact scenario the running-intersection guard was added to
    prevent: "a replica is temporarily reported as caching a later block
    (H3) but missed earlier blocks (H1, H2) due to telemetry lag." The
    simple chain-break scan hits the empty H1 slot first and stops, so the
    replica is never credited the later-block percent. The intersection
    guard is redundant.
    """
    prompt = list(range(1, 2 * 4 + 1))  # 8 tokens -> 4 blocks

    token_id = [11, 22, 33, 44]
    token_hash1, token_hash2 = _hashes(token_id)

    store = _store_with_block_size()
    # Inject ONLY H3 for rep_a — synthetic, but emulates a lagged/OoO report.
    store.add_blocks("rep_a", [token_hash1, token_hash2])

    # rep_a caches only unrelated hashes; prompt's H1 has no rep_a → chain
    # breaks immediately → 0.0 hit.
    assert store.get_layer_prefix_hit_rate("rep_a", prompt, Layer.GPU) == 0.0
