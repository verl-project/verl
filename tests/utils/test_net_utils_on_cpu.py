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
import socket
from unittest.mock import patch

import pytest

from verl.utils.net_utils import get_free_port_range


def test_get_free_port_range_count_one_alive():
    port, socks = get_free_port_range("127.0.0.1", 1, with_alive_socks=True)
    try:
        assert socks is not None and len(socks) == 1
        assert socks[0].getsockname()[1] == port
    finally:
        for sock in socks or []:
            sock.close()


def test_get_free_port_range_count_one_closed():
    port, socks = get_free_port_range("127.0.0.1", 1, with_alive_socks=False)
    assert isinstance(port, int) and socks is None


def test_get_free_port_range_consecutive():
    start, socks = get_free_port_range("127.0.0.1", 4, with_alive_socks=True)
    try:
        assert [s.getsockname()[1] for s in socks] == [start, start + 1, start + 2, start + 3]
        for port in range(start, start + 4):
            blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(OSError):
                blocker.bind(("127.0.0.1", port))
            blocker.close()
    finally:
        for sock in socks:
            sock.close()


def test_get_free_port_range_successive_ranges_are_disjoint():
    start_a, socks_a = get_free_port_range("127.0.0.1", 4, with_alive_socks=True)
    try:
        start_b, socks_b = get_free_port_range("127.0.0.1", 4, with_alive_socks=True)
        try:
            ports_a = {start_a + i for i in range(4)}
            ports_b = {s.getsockname()[1] for s in socks_b}
            assert ports_b == {start_b + i for i in range(4)}
            assert ports_a.isdisjoint(ports_b)
        finally:
            for sock in socks_b:
                sock.close()
    finally:
        for sock in socks_a:
            sock.close()


def test_get_free_port_range_forced_adjacent_collision_is_disjoint():
    start_a, socks_a = get_free_port_range("127.0.0.1", 4, with_alive_socks=True)
    socks_b = None
    try:
        if start_a < 2:
            pytest.skip("start_a too low to force an adjacent collision")
        occupied = {start_a + i for i in range(4)}
        real_bind = socket.socket.bind
        forced = {"bound": False}

        def bind_first_ephemeral_to_start_minus_one(self, addr):
            host, port = addr[:2]
            if port == 0 and not forced["bound"]:
                addr = (host, start_a - 1)
                result = real_bind(self, addr)
                forced["bound"] = True
                return result
            return real_bind(self, addr)

        with patch.object(socket.socket, "bind", bind_first_ephemeral_to_start_minus_one):
            start_b, socks_b = get_free_port_range("127.0.0.1", 4, with_alive_socks=True)
        if not forced["bound"]:
            pytest.fail("failed to force first bind onto start_a - 1")
        ports_b = {s.getsockname()[1] for s in socks_b}
        assert ports_b == {start_b + i for i in range(4)}
        assert ports_b.isdisjoint(occupied)
    finally:
        for sock in socks_a:
            sock.close()
        for sock in socks_b or []:
            sock.close()


def test_get_free_port_range_skips_occupied():
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    occupied = holder.getsockname()[1]
    try:
        start, socks = get_free_port_range("127.0.0.1", 2, with_alive_socks=True)
        try:
            ports = [s.getsockname()[1] for s in socks]
            assert occupied not in ports
            assert ports == [start, start + 1]
        finally:
            for sock in socks:
                sock.close()
    finally:
        holder.close()


@pytest.mark.parametrize("count", [0, -1])
def test_get_free_port_range_rejects_bad_count(count):
    with pytest.raises(ValueError):
        get_free_port_range("127.0.0.1", count)
