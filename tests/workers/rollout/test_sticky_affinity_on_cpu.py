# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""Bounded sticky affinity in ``GlobalRequestLoadBalancer``.

The balancer is a plain class wrapped with ``ray.remote`` at instantiation, so these tests
drive it directly -- no actor, no GPU.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

SERVERS = {"s0": None, "s1": None, "s2": None}


def lb(**kwargs) -> GlobalRequestLoadBalancer:
    kwargs.setdefault("servers", dict(SERVERS))
    return GlobalRequestLoadBalancer(**kwargs)


class TestDefaultBehaviorUnchanged:
    """The default must be indistinguishable from the pre-change balancer."""

    def test_least_loaded_round_robins(self):
        balancer = lb()
        picks = [balancer.acquire_server(f"r{i}")[0] for i in range(6)]
        assert sorted(picks) == ["s0", "s0", "s1", "s1", "s2", "s2"]

    def test_ties_resolve_to_first_candidate(self):
        assert lb().acquire_server("r0")[0] == "s0"

    def test_sticky_session_is_honored(self):
        balancer = lb()
        pinned = balancer.acquire_server("conv")[0]
        for _ in range(8):
            assert balancer.acquire_server("conv")[0] == pinned

    def test_affinity_never_breaks_by_default(self):
        balancer = lb()
        pinned = balancer.acquire_server("conv")[0]
        balancer._inflight_requests[pinned] += 1000
        assert balancer.acquire_server("conv")[0] == pinned
        assert balancer.get_status()["affinity_broken"] == 0

    def test_removed_server_still_reselects(self):
        balancer = lb()
        pinned = balancer.acquire_server("conv")[0]
        balancer.remove_servers([pinned])
        assert balancer.acquire_server("conv")[0] != pinned

    def test_no_servers_raises(self):
        with pytest.raises(RuntimeError, match="No available servers"):
            lb(servers={}).acquire_server("r0")

    def test_has_no_async_methods(self):
        # Any async method makes Ray create an asyncio actor, dropping the FIFO ordering
        # callers see today.
        coros = [
            name
            for name, fn in inspect.getmembers(GlobalRequestLoadBalancer, inspect.isfunction)
            if asyncio.iscoroutinefunction(fn)
        ]
        assert coros == [], coros


class TestBoundedAffinity:
    def test_overloaded_pin_is_abandoned(self):
        balancer = lb(affinity_break_margin=2)
        pinned = balancer.acquire_server("conv")[0]
        balancer._inflight_requests[pinned] += 10
        assert balancer.acquire_server("conv")[0] != pinned
        assert balancer.get_status()["affinity_broken"] == 1

    def test_pin_survives_a_gap_inside_the_margin(self):
        balancer = lb(affinity_break_margin=5)
        pinned = balancer.acquire_server("conv")[0]
        balancer._inflight_requests[pinned] += 4
        assert balancer.acquire_server("conv")[0] == pinned

    def test_boundary_is_inclusive(self):
        balancer = lb(affinity_break_margin=3)
        pinned = balancer.acquire_server("conv")[0]
        balancer._inflight_requests[pinned] = min(balancer._inflight_requests.values()) + 3
        assert balancer.acquire_server("conv")[0] == pinned
        balancer._inflight_requests[pinned] += 1
        assert balancer.acquire_server("conv")[0] != pinned

    def test_reroute_repins_so_later_turns_follow(self):
        balancer = lb(affinity_break_margin=0)
        balancer.acquire_server("conv")
        balancer._inflight_requests["s0"] = 50
        balancer._inflight_requests["s1"] = 0
        balancer._inflight_requests["s2"] = 50
        assert balancer.acquire_server("conv")[0] == "s1"
        # The new pin must hold, or every turn re-prefills somewhere new.
        balancer._inflight_requests["s1"] = 0
        assert balancer.acquire_server("conv")[0] == "s1"

    def test_zero_margin_tracks_the_least_loaded(self):
        balancer = lb(affinity_break_margin=0)
        balancer.acquire_server("conv")
        for target in ("s2", "s0", "s1"):
            for sid in balancer._inflight_requests:
                balancer._inflight_requests[sid] = 0 if sid == target else 9
            assert balancer.acquire_server("conv")[0] == target

    def test_first_placement_is_not_an_affinity_decision(self):
        balancer = lb(affinity_break_margin=2)
        balancer.acquire_server("conv")
        status = balancer.get_status()
        assert (status["affinity_kept"], status["affinity_broken"]) == (0, 0)
        balancer.acquire_server("conv")
        status = balancer.get_status()
        assert (status["affinity_kept"], status["affinity_broken"]) == (1, 0)

    def test_full_determinism_ignores_load(self):
        balancer = lb(affinity_break_margin=0, full_determinism=True)
        first = balancer.acquire_server("conv")[0]
        balancer._inflight_requests[first] += 100
        assert balancer.acquire_server("conv")[0] == first
        assert balancer.get_status()["affinity_broken"] == 0

    def test_release_lets_a_pin_become_acceptable_again(self):
        balancer = lb(affinity_break_margin=2)
        pinned = balancer.acquire_server("conv")[0]
        balancer._inflight_requests[pinned] += 10
        moved = balancer.acquire_server("conv")[0]
        assert moved != pinned
        for _ in range(10):
            balancer.release_server(pinned)
        # Balanced again, so the current pin holds rather than bouncing back.
        assert balancer.acquire_server("conv")[0] == moved

    @pytest.mark.parametrize("bad", [-1, -0.5, float("-inf"), float("nan")])
    def test_invalid_margins_are_rejected(self, bad):
        # These break affinity even when the pin IS the least loaded, so affinity_broken
        # would count moves that never happened.
        with pytest.raises(ValueError, match="affinity_break_margin"):
            lb(affinity_break_margin=bad)

    def test_counters_cover_every_sticky_hit(self):
        balancer = lb(affinity_break_margin=1)
        balancer.acquire_server("conv")
        for i in range(6):
            balancer._inflight_requests["s0"] = i
            balancer.acquire_server("conv")
        status = balancer.get_status()
        assert status["affinity_kept"] + status["affinity_broken"] == 6
