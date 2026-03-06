# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
from datetime import timedelta

import vllm
from torch.distributed import TCPStore
from vllm.distributed.utils import StatelessProcessGroup

from verl.utils.device import is_npu_available


@staticmethod
def create(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    data_expiration_seconds: int = 3600,
    store_timeout: int = 300,
) -> "StatelessProcessGroup":
    """A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state.

    If we have process A and process B called `torch.distributed.init_process_group`
    to form a group, and then we want to form another group with process A, B, C,
    D, it is not possible in PyTorch, because process A and process B have already
    formed a group, and process C and process D cannot join that group. This
    function is a workaround for this issue.

    `torch.distributed.init_process_group` is a global call, while this function
    is a stateless call. It will return a `StatelessProcessGroup` object that can be
    used for exchanging metadata. With this function, process A and process B
    can call `StatelessProcessGroup.create` to form a group, and then process A, B,
    C, and D can call `StatelessProcessGroup.create` to form another group.

    Args:
        host: Host address (IPv4 or IPv6). For IPv6, can be in format like "::1" or "[::1]".
        port: Port number to bind/listen on.
        rank: Rank of the current process.
        world_size: Total number of processes in the group.
        data_expiration_seconds: Time in seconds before data entries expire (default: 3600).
        store_timeout: Timeout in seconds for TCPStore connection (default: 300).

    Returns:
        StatelessProcessGroup: A stateless process group instance.
    """  # noqa
    # Detect address family (IPv4 or IPv6)
    try:
        # Try to parse as IPv6 first (IPv6 addresses are more specific)
        ipaddress.IPv6Address(host.strip("[]"))
        address_family = socket.AF_INET6
    except (ipaddress.AddressValueError, ValueError):
        address_family = socket.AF_INET

    launch_server = rank == 0
    if launch_server:
        # listen on the specified interface (instead of 0.0.0.0 or ::)
        listen_socket = socket.socket(address_family, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # For IPv6, set IPV6_V6ONLY to only listen on IPv6 (not dual-stack)
        # This ensures consistent behavior across different systems
        if address_family == socket.AF_INET6:
            try:
                listen_socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            except (AttributeError, OSError):
                # IPV6_V6ONLY might not be available on all systems
                pass

        # Remove brackets from IPv6 address if present (socket.bind handles it)
        bind_host = host.strip("[]")
        listen_socket.bind((bind_host, port))
        listen_socket.listen()
        listen_fd = listen_socket.fileno()
    else:
        listen_socket = None
        listen_fd = None

    store = TCPStore(
        host_name=host,
        port=port,
        world_size=world_size,
        is_master=launch_server,
        timeout=timedelta(seconds=store_timeout),
        use_libuv=False,  # for now: github.com/pytorch/pytorch/pull/150215
        master_listen_fd=listen_fd,
    )

    return StatelessProcessGroup(
        rank=rank,
        world_size=world_size,
        store=store,
        socket=listen_socket,
        data_expiration_seconds=data_expiration_seconds,
    )


vllm.distributed.utils.StatelessProcessGroup.create = create


def vllm_stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    # NOTE: If it is necessary to support weight synchronization with the sglang backend in the future,
    # the following can be used:
    # from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    # from sglang.srt.distributed.utils import statelessprocessgroup
    if is_npu_available:
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator as PyNcclCommunicator,
        )
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl
