from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Optional

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.utils.net_utils import is_valid_ipv6_address


def format_tcp_address(ip: str, port: int | str) -> str:
    host = f"[{ip}]" if is_valid_ipv6_address(ip) else ip
    return f"tcp://{host}:{port}"


@dataclass
class DecoupledSpecEndpointConfig:
    bind_endpoint: str
    connect_endpoints: list[str]
    rank: int

    def to_server_config(self, *, algorithm: str, speculative_num_steps: int, trace_dir: Optional[str] = None) -> dict:
        return {
            "algorithm": algorithm,
            "bind_endpoint": self.bind_endpoint,
            "connect_endpoints": self.connect_endpoints,
            "rank": self.rank,
            "speculative_num_steps": speculative_num_steps,
            "trace_dir": trace_dir,
        }


@dataclass
class DecoupledSpecTopology:
    verifier_configs: list[DecoupledSpecEndpointConfig]
    drafter_configs: list[DecoupledSpecEndpointConfig]


@ray.remote
class PortActor:
    def __init__(self):
        self._reserved_socket: Optional[socket.socket] = None

    def reserve_port(self, avoid_ports: Optional[list[int]] = None) -> dict[str, Any]:
        self.release_port()
        avoid_ports = set(avoid_ports or [])

        for _ in range(256):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("0.0.0.0", 0))
            probe.listen(1)
            port = int(probe.getsockname()[1])
            probe.close()
            if port in avoid_ports:
                continue

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                sock.listen(1)
            except OSError:
                sock.close()
                continue

            self._reserved_socket = sock
            return {"host": ray.util.get_node_ip_address(), "port": port}

        raise RuntimeError("failed to reserve a decoupled-spec TCP port")

    def release_port(self) -> bool:
        if self._reserved_socket is not None:
            self._reserved_socket.close()
            self._reserved_socket = None
        return True


def _extract_port(endpoint: str) -> Optional[int]:
    try:
        return int(endpoint.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def reserve_endpoint_on_node(node_id: str, used_ports: set[int]) -> str:
    actor = PortActor.options(
        num_cpus=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
    ).remote()
    try:
        reservation = ray.get(actor.reserve_port.remote(sorted(used_ports)))
        endpoint = format_tcp_address(reservation["host"], int(reservation["port"]))
        ray.get(actor.release_port.remote())
    finally:
        ray.kill(actor, no_restart=True)

    port = _extract_port(endpoint)
    if port is not None:
        used_ports.add(port)
    return endpoint


def create_decoupled_spec_topology(
    verifier_runtime_infos: list[dict[str, str]],
    drafter_runtime_infos: list[dict[str, str]],
) -> DecoupledSpecTopology:
    if not verifier_runtime_infos:
        raise ValueError("decoupled spec requires at least one verifier replica")
    if not drafter_runtime_infos:
        raise ValueError("decoupled spec requires at least one drafter replica")

    used_ports: set[int] = set()
    verifier_result_endpoints = [
        reserve_endpoint_on_node(info["node_id"], used_ports) for info in verifier_runtime_infos
    ]
    drafter_control_endpoints = [
        reserve_endpoint_on_node(info["node_id"], used_ports) for info in drafter_runtime_infos
    ]

    verifier_configs = [
        DecoupledSpecEndpointConfig(
            bind_endpoint=endpoint,
            connect_endpoints=drafter_control_endpoints,
            rank=rank,
        )
        for rank, endpoint in enumerate(verifier_result_endpoints)
    ]
    drafter_configs = [
        DecoupledSpecEndpointConfig(
            bind_endpoint=endpoint,
            connect_endpoints=verifier_result_endpoints,
            rank=rank,
        )
        for rank, endpoint in enumerate(drafter_control_endpoints)
    ]
    return DecoupledSpecTopology(verifier_configs=verifier_configs, drafter_configs=drafter_configs)

