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


def test_is_ipv6():
    assert is_ipv6("::1")
    assert not is_ipv6("127.0.0.1")
    # `ipaddress` does not accept the bracketed URL form, which is why callers
    # that add brackets rely on this returning False.
    assert not is_ipv6("[::1]")


def test_is_valid_ipv6_address_matches_is_ipv6_for_bracketed_form():
    assert is_valid_ipv6_address("[::1]") is False


def test_get_free_port_ipv4():
    port, sock = get_free_port("127.0.0.1")
    assert isinstance(port, int)
    assert port > 0
    assert sock is None


def test_get_free_port_keeps_socket_alive_when_requested():
    port, sock = get_free_port("127.0.0.1", with_alive_sock=True)
    assert sock is not None
    assert sock.getsockname()[1] == port
    sock.close()


@_ipv6
def test_get_free_port_ipv6():
    port, sock = get_free_port("::1")
    assert isinstance(port, int)
    assert sock is None


@_ipv6
def test_get_free_port_bracketed_ipv6():
    """The bracketed form is what `ray.util.get_node_ip_address()` returns for
    IPv6, and what URLs use. It must be accepted rather than bound as AF_INET.
    """
    port, sock = get_free_port("[::1]")
    assert isinstance(port, int)
    assert port > 0
    assert sock is None


@_ipv6
def test_get_free_port_bracketed_ipv6_matches_unbracketed_family():
    """Both forms must land on AF_INET6, so the reserved socket's family is the
    only thing that needs checking here.
    """
    _, sock = get_free_port("[::1]", with_alive_sock=True)
    try:
        assert sock.family == socket.AF_INET6
    finally:
        sock.close()
