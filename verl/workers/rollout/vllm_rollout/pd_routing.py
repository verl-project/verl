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

"""Decode routing policies for vLLM prefill-decode disaggregation.

The policy names, configuration fields, defaults, and selection semantics are
adapted from ``vllm-project/router``. Verl routes token IDs between Ray actors,
so cache-aware routing indexes token prefixes instead of raw prompt text and
uses actor-local in-flight counters as the vLLM Router ``Worker.load()`` value.
"""

from __future__ import annotations

import bisect
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from verl.workers.config.disaggregation import RoutingPolicyConfig

__all__ = ["DecodePeerSelector"]


class _Sampler(Protocol):
    def sample(self, population: Sequence[int], k: int) -> list[int]: ...


@dataclass(eq=False)
class _RadixNode:
    text: tuple[int, ...]
    parent: _RadixNode | None
    children: dict[int, _RadixNode] = field(default_factory=dict)
    tenant_access: dict[int, int] = field(default_factory=dict)


class _TokenRadixTree:
    """Compressed, multi-tenant radix tree adapted from vLLM Router's Tree."""

    def __init__(self, num_tenants: int):
        self._epoch = 0
        self.root = _RadixNode((), None, tenant_access={tenant: 0 for tenant in range(num_tenants)})
        self.tenant_sizes = [0] * num_tenants

    @staticmethod
    def _shared_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
        return next(
            (index for index, pair in enumerate(zip(left, right, strict=False)) if pair[0] != pair[1]),
            min(len(left), len(right)),
        )

    def _next_epoch(self) -> int:
        self._epoch += 1
        return self._epoch

    def insert(self, token_ids: Sequence[int], tenant: int) -> None:
        tokens = tuple(token_ids)
        epoch = self._next_epoch()
        if not tokens:
            self.root.tenant_access[tenant] = epoch
            return

        parent = self.root
        offset = 0
        while offset < len(tokens):
            first_token = tokens[offset]
            child = parent.children.get(first_token)
            if child is None:
                segment = tokens[offset:]
                parent.children[first_token] = _RadixNode(segment, parent, tenant_access={tenant: epoch})
                self.tenant_sizes[tenant] += len(segment)
                return

            shared = self._shared_prefix_len(tokens[offset:], child.text)
            if shared < len(child.text):
                prefix = child.text[:shared]
                suffix = child.text[shared:]
                split = _RadixNode(prefix, parent, {suffix[0]: child}, dict(child.tenant_access))
                parent.children[first_token] = split
                child.text = suffix
                child.parent = split

                if tenant not in split.tenant_access:
                    split.tenant_access[tenant] = 0
                    self.tenant_sizes[tenant] += len(prefix)

                offset += shared
                if offset == len(tokens):
                    split.tenant_access[tenant] = epoch
                    return

                segment = tokens[offset:]
                split.children[segment[0]] = _RadixNode(segment, split, tenant_access={tenant: epoch})
                self.tenant_sizes[tenant] += len(segment)
                return

            if tenant not in child.tenant_access:
                child.tenant_access[tenant] = 0
                self.tenant_sizes[tenant] += len(child.text)
            offset += shared
            parent = child

        parent.tenant_access[tenant] = epoch

    def prefix_match(self, token_ids: Sequence[int]) -> tuple[int, int]:
        node = self.root
        offset = 0
        while offset < len(token_ids):
            child = node.children.get(token_ids[offset])
            if child is None:
                break
            shared = self._shared_prefix_len(token_ids[offset:], child.text)
            offset += shared
            node = child
            if shared < len(child.text):
                break
        return max(node.tenant_access, key=node.tenant_access.get), offset

    def evict_tenant_by_size(self, tenant: int, max_size: int) -> None:
        while self.tenant_sizes[tenant] > max_size:
            leaves: list[_RadixNode] = []
            stack = list(self.root.children.values())
            while stack:
                node = stack.pop()
                stack.extend(node.children.values())
                if tenant in node.tenant_access and not any(
                    tenant in child.tenant_access for child in node.children.values()
                ):
                    leaves.append(node)
            if not leaves:
                return

            leaf = min(leaves, key=lambda node: node.tenant_access[tenant])
            del leaf.tenant_access[tenant]
            self.tenant_sizes[tenant] = max(0, self.tenant_sizes[tenant] - len(leaf.text))
            if not leaf.tenant_access and not leaf.children and leaf.parent is not None:
                del leaf.parent.children[leaf.text[0]]


def _murmur_hash_64a(key: bytes, seed: int = 4193360111) -> int:
    mask = (1 << 64) - 1
    multiplier = 0xC6A4A7935BD1E995
    shift = 47
    value = (seed ^ (len(key) * multiplier)) & mask
    chunk_end = len(key) - len(key) % 8
    for offset in range(0, chunk_end, 8):
        chunk = int.from_bytes(key[offset : offset + 8], "little")
        chunk = (chunk * multiplier) & mask
        chunk ^= chunk >> shift
        chunk = (chunk * multiplier) & mask
        value ^= chunk
        value = (value * multiplier) & mask

    remainder = key[chunk_end:]
    for index, byte in enumerate(remainder):
        value ^= byte << (index * 8)
    if remainder:
        value = (value * multiplier) & mask
    value ^= value >> shift
    value = (value * multiplier) & mask
    value ^= value >> shift
    return value & mask


def _murmur_rehash_64a(key: int) -> int:
    return _murmur_hash_64a(key.to_bytes(8, "little"))


def _furc_get_bit(key: bytes, index: int, hash_cache: list[int]) -> bool:
    order = index >> 6
    while len(hash_cache) <= order:
        hash_cache.append(_murmur_hash_64a(key) if not hash_cache else _murmur_rehash_64a(hash_cache[-1]))
    return bool((hash_cache[order] >> (index & 0x3F)) & 1)


def _furc_hash(key: bytes, modulus: int) -> int:
    if modulus <= 1:
        return 0
    depth = 0
    while modulus > 1 << depth:
        depth += 1
    index = depth
    hash_cache: list[int] = []
    for _ in range(32):
        while not _furc_get_bit(key, index, hash_cache):
            if depth == 0:
                return 0
            depth -= 1
            index = depth
        index += 23
        value = 1
        for _ in range(max(0, depth - 1)):
            value = (value << 1) | _furc_get_bit(key, index, hash_cache)
            index += 23
        if value < modulus:
            return value
    return 0


def _fbi_hash(key: str) -> int:
    furc_result = _furc_hash(key.encode(), (1 << 23) - 1)
    return _murmur_hash_64a(furc_result.to_bytes(4, "little"))


def _policy_config(config: RoutingPolicyConfig | Mapping[str, Any]) -> RoutingPolicyConfig:
    return config if isinstance(config, RoutingPolicyConfig) else RoutingPolicyConfig(**dict(config))


class DecodePeerSelector:
    """Select a decode peer with vLLM Router-compatible policy semantics."""

    def __init__(
        self,
        policy_config: RoutingPolicyConfig | Mapping[str, Any],
        peer_ids: Sequence[str],
        sampler: _Sampler = random,
    ):
        if not peer_ids:
            raise ValueError("peer_ids must not be empty")
        if len(set(peer_ids)) != len(peer_ids):
            raise ValueError("peer_ids must be unique")

        self.config = _policy_config(policy_config)
        self.peer_ids = tuple(peer_ids)
        self.pending_requests = [0] * len(peer_ids)
        self._sampler = sampler
        self._round_robin_cursor = 0
        self._hash_ring = (
            sorted(
                (_fbi_hash(f"{peer_id}:{virtual_node}"), index)
                for index, peer_id in enumerate(self.peer_ids)
                for virtual_node in range(self.config.virtual_nodes)
            )
            if self.config.type == "consistent_hash"
            else []
        )
        self._hash_points = [point for point, _ in self._hash_ring]
        self._cache_tree = _TokenRadixTree(len(peer_ids)) if self.config.type == "cache_aware" else None
        self._last_eviction = time.monotonic()

    def acquire(
        self,
        *,
        routing_key: str | None = None,
        prompt_ids: Sequence[int] | None = None,
        eligible: Sequence[int] | None = None,
    ) -> int:
        index = self.select(routing_key=routing_key, prompt_ids=prompt_ids, eligible=eligible)
        self.pending_requests[index] += 1
        return index

    def select(
        self,
        *,
        routing_key: str | None = None,
        prompt_ids: Sequence[int] | None = None,
        eligible: Sequence[int] | None = None,
    ) -> int:
        candidates = list(range(len(self.pending_requests))) if eligible is None else list(eligible)
        self._validate_candidates(candidates)

        policy = self.config.type
        if policy == "random":
            return self._sampler.sample(candidates, 1)[0]
        if policy == "round_robin":
            selected = candidates[self._round_robin_cursor % len(candidates)]
            self._round_robin_cursor += 1
            return selected
        if policy == "power_of_two":
            return self._select_power_of_two(candidates)
        if policy in ("consistent_hash", "rendezvous_hash"):
            if routing_key is None:
                raise ValueError(f"{policy} requires routing_key")
            if policy == "consistent_hash":
                return self._select_consistent_hash(candidates, routing_key)
            return max(candidates, key=lambda index: _fbi_hash(f"{routing_key}:{self.peer_ids[index]}"))
        if prompt_ids is None:
            raise ValueError("cache_aware requires prompt_ids")
        return self._select_cache_aware(candidates, prompt_ids)

    def _validate_candidates(self, candidates: Sequence[int]) -> None:
        if not candidates:
            raise RuntimeError("no eligible decode peer")
        if len(set(candidates)) != len(candidates):
            raise ValueError("eligible decode peer indices must be unique")
        if any(index < 0 or index >= len(self.pending_requests) for index in candidates):
            raise ValueError(f"eligible decode peer index out of range: {candidates}")

    def _select_power_of_two(self, candidates: Sequence[int]) -> int:
        if len(candidates) == 1:
            return candidates[0]
        first, second = self._sampler.sample(candidates, 2)
        return first if self.pending_requests[first] <= self.pending_requests[second] else second

    def _select_consistent_hash(self, candidates: Sequence[int], routing_key: str) -> int:
        if not self._hash_ring:
            raise RuntimeError("consistent-hash ring is not initialized")
        position = bisect.bisect_left(self._hash_points, _fbi_hash(routing_key))
        if position == len(self._hash_ring):
            position = 0
        selected = self._hash_ring[position][1]
        return selected if selected in candidates else candidates[0]

    def _select_cache_aware(self, candidates: Sequence[int], prompt_ids: Sequence[int]) -> int:
        if self._cache_tree is None:
            raise RuntimeError("cache-aware radix tree is not initialized")
        self._evict_cache_if_due()
        candidate_loads = [self.pending_requests[index] for index in candidates]
        min_load = min(candidate_loads)
        max_load = max(candidate_loads)
        imbalanced = (
            max_load - min_load > self.config.balance_abs_threshold
            and max_load > min_load * self.config.balance_rel_threshold
        )

        if imbalanced:
            selected = min(candidates, key=self.pending_requests.__getitem__)
        else:
            matched_tenant, matched_tokens = self._cache_tree.prefix_match(prompt_ids)
            match_rate = matched_tokens / len(prompt_ids) if prompt_ids else 0.0
            if match_rate > self.config.cache_threshold and matched_tenant in candidates:
                selected = matched_tenant
            elif match_rate > self.config.cache_threshold:
                selected = candidates[0]
            else:
                selected = min(candidates, key=self.pending_requests.__getitem__)

        self._cache_tree.insert(prompt_ids, selected)
        return selected

    def _evict_cache_if_due(self) -> None:
        if self._cache_tree is None:
            return
        interval = self.config.eviction_interval_secs
        now = time.monotonic()
        if interval <= 0 or now - self._last_eviction < interval:
            return
        for tenant in range(len(self.pending_requests)):
            self._cache_tree.evict_tenant_by_size(tenant, self.config.max_tree_size)
        self._last_eviction = now

    def release(self, index: int) -> None:
        if self.pending_requests[index] < 1:
            raise RuntimeError(f"decode peer {index} has no in-flight request to release")
        self.pending_requests[index] -= 1

    def clear_cache(self) -> None:
        if self._cache_tree is not None:
            self._cache_tree = _TokenRadixTree(len(self.pending_requests))
            self._last_eviction = time.monotonic()
