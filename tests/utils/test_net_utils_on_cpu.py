# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import pytest

from verl.utils.net_utils import (
    get_free_port,
    is_ipv4,
    is_ipv6,
    is_valid_ipv6_address,
)


def _ipv6_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
        return True
    except OSError:
        return False


_ipv6 = pytest.mark.skipif(not _ipv6_available(), reason="IPv6 loopback unavailable")


def test_is_ipv4():
    assert is_ipv4("127.0.0.1")
    assert is_ipv4("0.0.0.0")
    assert not is_ipv4("::1")
    assert not is_ipv4("localhost")


def test_is_ipv4_rejects_out_of_range_octets():
    assert not is_ipv4("256.0.0.1")
    # `ipaddress` rejects leading zeros rather than silently accepting them.
    assert not is_ipv4("010.0.0.1")


def test_is_ipv6():
    assert is_ipv6("::1")
    assert is_ipv6("2001:db8::1")
    assert not is_ipv6("127.0.0.1")
    assert not is_ipv6("localhost")


def test_bracketed_ipv6_is_not_a_valid_address():
    """Callers must strip the brackets themselves.

    Every `get_free_port` call site normalises with
    `ray.util.get_node_ip_address().strip("[]")` before calling, because
    `ipaddress` rejects the bracketed form that URLs (and Ray for IPv6) use.
    These assertions pin that contract: if a future change made the bracketed
    form valid, the call sites that *add* brackets for URL hosts
    (`f"[{addr}]:{port}" if is_valid_ipv6_address(addr)`) would start
    double-bracketing.
    """
    assert not is_ipv6("[::1]")
    assert not is_valid_ipv6_address("[::1]")


def test_get_free_port_ipv4():
    port, sock = get_free_port("127.0.0.1")
    assert isinstance(port, int)
    assert port > 0
    assert sock is None


def test_get_free_port_returns_a_bound_reserved_socket():
    port, sock = get_free_port("127.0.0.1", with_alive_sock=True)
    try:
        assert sock is not None
        assert sock.family == socket.AF_INET
        assert sock.getsockname()[1] == port
    finally:
        sock.close()


@_ipv6
def test_get_free_port_ipv6_uses_inet6_family():
    port, sock = get_free_port("::1", with_alive_sock=True)
    try:
        assert sock.family == socket.AF_INET6
        assert sock.getsockname()[1] == port
    finally:
        sock.close()


def test_get_free_port_hostname():
    port, sock = get_free_port("localhost")
    assert isinstance(port, int)
    assert sock is None
