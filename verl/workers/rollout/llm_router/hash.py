"""
Chained xxhash prefix hash computation.
"""

from __future__ import annotations

import struct

import xxhash


def compute_hash(parent_hash: int, block_bytes: bytes, seed: int = 0) -> int:
    """Compute xxhash for a single block given parent hash and token bytes.

    Algorithm (matching aibrix ``SyncPrefixHashTable.computeHash``):
        h = xxhash.xxh64(seed=seed)
        h.update(parent_hash as 8-byte little-endian)
        h.update(block_bytes)

    The parent hash is **always** written — for the first block in a
    chain, ``parent_hash`` should be ``seed`` (typically ``0``).

    Args:
        parent_hash: Parent (predecessor) prefix hash as ``int``.
            Use ``seed`` for the first block in a chain.
        block_bytes: Token bytes for this block, already encoded as
            uint32 big-endian (4 bytes per token).
        seed: xxhash seed value. Defaults to ``0``.

    Returns:
        Prefix hash as ``int`` (equivalent to Go ``uint64``).
    """
    h = xxhash.xxh64(seed=seed)
    h.update(parent_hash.to_bytes(8, "little"))
    h.update(block_bytes)
    return h.intdigest()


def get_prefix_hashes(
    prompt_ids: list[int],
    block_size: int,
    seed: int = 0,
) -> list[int]:
    """Compute prefix hash sequence for prompt token IDs.

    Algorithm (matching aibrix ``SyncPrefixHashTable.GetPrefixHashes``):
        parent_hash = seed
        for each full block:
            block_bytes = encode tokens as uint32 big-endian
            current_hash = compute_hash(parent_hash, block_bytes, seed)
            parent_hash = current_hash

    Only full blocks (exactly ``block_size`` tokens) are hashed.
    Partial blocks at the tail are ignored.

    Args:
        prompt_ids: Prompt token IDs as ``list[int]``.
        block_size: Number of tokens per block (must be > 0).
        seed: xxhash seed value. Defaults to ``0``.

    Returns:
        List of prefix hashes as ``int`` (one per full block).
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    n_full_blocks = len(prompt_ids) // block_size
    hashes: list[int] = []
    parent_hash = seed

    for i in range(n_full_blocks):
        start = i * block_size
        end = start + block_size
        block_tokens = prompt_ids[start:end]
        block_bytes = struct.pack(f">{block_size}I", *block_tokens)
        current_hash = compute_hash(parent_hash, block_bytes, seed)
        hashes.append(current_hash)
        parent_hash = current_hash

    return hashes
