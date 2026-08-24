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

import pytest

from verl.workers.config import RoutingPolicyConfig
from verl.workers.rollout.vllm_rollout.pd_routing import DecodePeerSelector


class _FixedSampler:
    def __init__(self, samples):
        self._samples = iter(samples)

    def sample(self, population, k):
        sample = next(self._samples)
        assert k == len(sample)
        assert all(index in population for index in sample)
        return list(sample)


def test_random_selects_from_eligible_peers_only():
    selector = DecodePeerSelector(
        RoutingPolicyConfig(type="random"),
        ["d0", "d1", "d2"],
        sampler=_FixedSampler([(2,)]),
    )

    assert selector.select(eligible=[1, 2]) == 2


def test_power_of_two_selects_lower_pending_peer_and_reserves():
    selector = DecodePeerSelector(
        RoutingPolicyConfig(type="power_of_two"),
        ["d0", "d1", "d2"],
        sampler=_FixedSampler([(0, 2)]),
    )
    selector.pending_requests[:] = [4, 0, 1]

    index = selector.acquire()

    assert index == 2
    assert selector.pending_requests == [4, 0, 2]


def test_power_of_two_only_samples_eligible_peers():
    selector = DecodePeerSelector(
        RoutingPolicyConfig(type="power_of_two"),
        ["d0", "d1", "d2", "d3"],
        sampler=_FixedSampler([(1, 3)]),
    )
    selector.pending_requests[:] = [0, 3, 0, 1]

    assert selector.select(eligible=[1, 3]) == 3


def test_power_of_two_tie_selects_first_sample_like_vllm_router():
    selector = DecodePeerSelector(
        RoutingPolicyConfig(type="power_of_two"),
        ["d0", "d1"],
        sampler=_FixedSampler([(0, 1), (0, 1)]),
    )

    assert [selector.select(), selector.select()] == [0, 0]


def test_decode_selector_release_and_empty_eligible_guard():
    selector = DecodePeerSelector(RoutingPolicyConfig(type="power_of_two"), ["d0"])
    selector.acquire()
    selector.release(0)

    assert selector.pending_requests == [0]
    with pytest.raises(RuntimeError, match="no eligible"):
        selector.select(eligible=[])
    with pytest.raises(RuntimeError, match="no in-flight"):
        selector.release(0)


def test_consistent_hash_keeps_session_on_same_peer():
    selector = DecodePeerSelector(RoutingPolicyConfig(type="consistent_hash"), ["d0", "d1", "d2", "d3"])

    picks = [selector.select(routing_key="episode-123") for _ in range(10)]

    assert len(set(picks)) == 1


def test_consistent_hash_falls_back_to_first_eligible_peer():
    selector = DecodePeerSelector(RoutingPolicyConfig(type="consistent_hash"), ["d0", "d1", "d2"])
    target = selector.select(routing_key="episode-123")
    eligible = [index for index in range(3) if index != target]

    assert selector.select(eligible=eligible, routing_key="episode-123") == eligible[0]


def test_consistent_hash_honors_virtual_nodes_config():
    selector = DecodePeerSelector(
        RoutingPolicyConfig(type="consistent_hash", virtual_nodes=7),
        ["http://d0:8000", "http://d1:8000"],
    )

    assert len(selector._hash_ring) == 14


def test_rendezvous_hash_keeps_session_on_same_peer():
    selector = DecodePeerSelector(
        RoutingPolicyConfig(type="rendezvous_hash"),
        ["http://d0:8000", "http://d1:8000", "http://d2:8000"],
    )

    picks = [selector.select(routing_key="episode-123") for _ in range(10)]

    assert len(set(picks)) == 1


def test_cache_aware_reuses_longest_prefix_when_load_is_balanced():
    selector = DecodePeerSelector(
        RoutingPolicyConfig(type="cache_aware", eviction_interval_secs=0),
        ["d0", "d1"],
    )
    prompt = list(range(100))

    first = selector.select(prompt_ids=prompt)

    assert selector.select(prompt_ids=prompt + [100]) == first


def test_cache_aware_uses_shortest_queue_when_load_is_imbalanced():
    selector = DecodePeerSelector(
        RoutingPolicyConfig(
            type="cache_aware",
            balance_abs_threshold=1,
            balance_rel_threshold=1.1,
            eviction_interval_secs=0,
        ),
        ["d0", "d1"],
    )
    prompt = list(range(100))
    assert selector.select(prompt_ids=prompt) == 0
    selector.pending_requests[:] = [5, 0]

    assert selector.select(prompt_ids=prompt) == 1
