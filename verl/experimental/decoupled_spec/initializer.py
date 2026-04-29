from __future__ import annotations

import math
from typing import Any

import ray
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.experimental.decoupled_spec.config import DraftConfig, get_draft_config
from verl.experimental.decoupled_spec.draft_server import DraftSGLangServer


def _get_alive_gpu_nodes() -> list[dict[str, Any]]:
    available_resources = ray._private.state.available_resources_per_node()
    node_infos = {node["NodeID"]: node for node in ray.nodes() if node.get("Alive")}

    candidates = []
    for node_id, node in node_infos.items():
        node_resources = available_resources.get(node_id, {})
        available_gpus = int(node_resources.get("GPU", node_resources.get("NPU", 0)))
        if available_gpus <= 0:
            continue
        candidates.append(
            {
                "node_id": node_id,
                "node_ip": node["NodeManagerAddress"],
                "available_gpus": available_gpus,
            }
        )

    candidates.sort(key=lambda item: (-item["available_gpus"], item["node_ip"]))
    return candidates


def _plan_draft_placement(
    draft_config: DraftConfig,
    *,
    n_gpus_per_node: int,
) -> list[str]:
    if draft_config.tp_size > n_gpus_per_node:
        raise ValueError(
            f"draft.tp_size ({draft_config.tp_size}) must be <= rollout.n_gpus_per_node ({n_gpus_per_node})"
        )
    if draft_config.ngpus > draft_config.nnodes * n_gpus_per_node:
        raise ValueError(
            "draft.ngpus must be <= draft.nnodes * rollout.n_gpus_per_node "
            f"(got {draft_config.ngpus} > {draft_config.nnodes} * {n_gpus_per_node})"
        )

    candidate_nodes = _get_alive_gpu_nodes()
    if len(candidate_nodes) < draft_config.nnodes:
        raise ValueError(
            f"Need at least {draft_config.nnodes} alive GPU nodes for draft sidecar, got {len(candidate_nodes)}"
        )

    selected_nodes = candidate_nodes[: draft_config.nnodes]
    total_capacity = sum((node["available_gpus"] // draft_config.tp_size) * draft_config.tp_size for node in selected_nodes)
    if total_capacity < draft_config.ngpus:
        raise ValueError(
            "Not enough free GPUs on selected draft nodes: "
            f"need {draft_config.ngpus}, capacity is {total_capacity}"
        )

    remaining = draft_config.num_drafters
    node_assignments: list[str] = []
    for node in selected_nodes:
        capacity = node["available_gpus"] // draft_config.tp_size
        take = min(capacity, remaining)
        node_assignments.extend([node["node_id"]] * take)
        remaining -= take
        if remaining == 0:
            break

    if remaining != 0:
        raise ValueError(f"Unable to place {draft_config.num_drafters} drafters with tp_size={draft_config.tp_size}")

    return node_assignments


def initialize_draft_servers(
    *,
    config: DictConfig,
    rollout_config,
    model_config,
) -> list[ray.actor.ActorHandle]:
    draft_config = get_draft_config(config)
    if not draft_config.enable:
        if rollout_config.get("enable_decoupled_spec", False):
            raise ValueError("draft.enable must be True when rollout.enable_decoupled_spec=True")
        return []

    n_gpus_per_node = int(rollout_config.n_gpus_per_node)
    node_assignments = _plan_draft_placement(draft_config, n_gpus_per_node=n_gpus_per_node)

    draft_server_actor_cls = ray.remote(DraftSGLangServer)
    server_handles = []
    for server_index, node_id in enumerate(node_assignments):
        server = draft_server_actor_cls.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            num_gpus=draft_config.tp_size,
            max_concurrency=max(8, math.ceil(draft_config.tp_size) + 4),
        ).remote(
            rollout_config=rollout_config,
            model_config=model_config,
            draft_config=draft_config,
            server_index=server_index,
        )
        server_handles.append(server)

    return server_handles
