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
"""The subset of torch.distributed._transport used by the publisher and puller, on NIXL's own API.

torch.distributed._transport needs torch >= 2.15, while released vLLM (0.33) still pins
torch 2.14. This adapter gives the same one-sided reads on any torch version, through the NIXL
Python bindings that the `nixl` checkpoint engine already uses.
"""

import asyncio
import pickle
import uuid

import nixl._api as nixl_api
import torch


class _RemoteBuffer:
    def __init__(self, agent_name: str, descs):
        self.agent_name, self.descs = agent_name, descs

    def serialize(self) -> bytes:
        return pickle.dumps((self.agent_name, self.descs))

    @classmethod
    def deserialize(cls, data: bytes) -> "_RemoteBuffer":
        return cls(*pickle.loads(data))


class _Memory:
    def __init__(self, transport: "NixlNativeTransport", tensor: torch.Tensor):
        self.transport = transport
        self.tensor = tensor
        self.reg_descs = transport.agent.register_memory(tensor)
        self.xfer_descs = transport.agent.get_xfer_descs(tensor)

    def to_remote_buffer(self) -> _RemoteBuffer:
        return _RemoteBuffer(self.transport.name, self.xfer_descs)

    def to_mutable_view(self) -> "_Memory":
        return self


class NixlNativeTransport:
    """One NIXL agent; `bind` returns its metadata, so register memory before calling it."""

    def __init__(self, device: str | None = None):
        self.name = uuid.uuid4().hex
        self.agent = nixl_api.nixl_agent(self.name)
        self.peers: list[str] = []

    def bind(self) -> bytes:
        return self.agent.get_agent_metadata()

    def connect(self, peer_url: bytes) -> None:
        self.peers.append(self.agent.add_remote_agent(peer_url).decode("utf-8"))

    def register_memory(self, tensor: torch.Tensor) -> _Memory:
        return _Memory(self, tensor)

    def unregister_memory(self, memory: _Memory) -> None:
        self.agent.deregister_memory(memory.reg_descs)

    async def read_async(self, local_buffer: _Memory, remote_buffer: _RemoteBuffer) -> None:
        handle = self.agent.initialize_xfer(
            "READ", local_buffer.xfer_descs, remote_buffer.descs, remote_buffer.agent_name, b""
        )
        try:
            state = self.agent.transfer(handle)
            while state != "DONE":
                if state == "ERR":
                    raise RuntimeError(f"NIXL read from {remote_buffer.agent_name} failed")
                await asyncio.sleep(0)
                state = self.agent.check_xfer_state(handle)
        finally:
            self.agent.release_xfer_handle(handle)

    def close(self) -> None:
        for peer in self.peers:
            self.agent.remove_remote_agent(peer)
        self.peers = []
