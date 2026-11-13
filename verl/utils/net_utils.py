# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
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
import ipaddress
import socket


def is_ipv4(ip_str: str) -> bool:
    """
    Check if the given string is an IPv4 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv4 address, False otherwise
    """
    try:
        ipaddress.IPv4Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_ipv6(ip_str: str) -> bool:
    """
    Check if the given string is an IPv6 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv6 address, False otherwise
    """
    try:
        ipaddress.IPv6Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str, with_alive_sock: bool = False) -> tuple[int, socket.socket | None]:
    """Find a free port on the given address.

    By default the socket is closed internally, suitable for immediate use.
    Set with_alive_sock=True to keep the socket open as a port reservation,
    preventing other calls from getting the same port. The caller is
    responsible for closing the socket before the port is actually bound
    by the target service (e.g. NCCL, uvicorn).
    """
    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((address, 0))
    port = sock.getsockname()[1]
    if with_alive_sock:
        return port, sock
    sock.close()
    return port, None


def get_free_port_range(
    address: str,
    count: int,
    with_alive_socks: bool = False,
    max_tries: int = 100,
) -> tuple[int, list[socket.socket] | None]:
    """Find `count` consecutive free TCP ports on `address`.

    When count==1, delegates to get_free_port. Otherwise the first port is
    bind((address, 0)) with SO_REUSEADDR. For start+1..start+count-1, probe
    without SO_REUSEADDR first (Linux otherwise allows overlapping REUSEADDR
    binds on bound-but-not-listening sockets), then hold the port with
    SO_REUSEADDR so the caller can bind before closing the reservation.
    Set with_alive_socks=True to keep the sockets open; the caller must close them.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if count == 1:
        port, sock = get_free_port(address, with_alive_sock=with_alive_socks)
        return port, [sock] if sock is not None else None

    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET
    for _ in range(max_tries):
        socks: list[socket.socket] = []
        try:
            sock = socket.socket(family=family, type=socket.SOCK_STREAM)
            socks.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((address, 0))
            start = sock.getsockname()[1]
            if start + count - 1 > 65535:
                raise OSError("port range exceeds 65535")
            for offset in range(1, count):
                # Probe without SO_REUSEADDR: Linux allows overlapping REUSEADDR binds on
                # bound-but-not-listening sockets, which would collide PD handshake ranges.
                probe = socket.socket(family=family, type=socket.SOCK_STREAM)
                try:
                    probe.bind((address, start + offset))
                finally:
                    probe.close()
                extra = socket.socket(family=family, type=socket.SOCK_STREAM)
                socks.append(extra)
                extra.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                extra.bind((address, start + offset))
        except OSError:
            for sock in socks:
                sock.close()
            continue
        if with_alive_socks:
            return start, socks
        for sock in socks:
            sock.close()
        return start, None
    raise RuntimeError(f"Failed to find {count} consecutive free ports on {address}")
