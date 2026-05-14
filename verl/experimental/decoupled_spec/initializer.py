from __future__ import annotations

import asyncio
from typing import Any

import ray
from omegaconf import DictConfig

from verl.experimental.decoupled_spec.config import (
    DraftConfig,
    build_draft_model_config,
    build_draft_rollout_config,
    get_draft_config,
    validate_decoupled_spec_config,
)
from verl.experimental.decoupled_spec.topology import create_decoupled_spec_topology
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import split_resource_pool
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica


def _create_verifier_replicas(
    *,
    rollout_replica_class: type,
    rollout_config: RolloutConfig,
    model_config: HFModelConfig,
    num_replicas: int,
) -> list[Any]:
    return [
        rollout_replica_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=rollout_config.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]


def _create_drafter_replicas(
    *,
    rollout_config: RolloutConfig,
    model_config: HFModelConfig,
    draft_config: DraftConfig,
) -> tuple[list[SGLangReplica], list[RayResourcePool]]:
    _check_draft_resources_available(draft_config)
    draft_rollout_config = build_draft_rollout_config(rollout_config, draft_config)
    draft_model_config = build_draft_model_config(model_config, draft_config)
    draft_resource_pool = RayResourcePool(
        process_on_nodes=[draft_config.ngpus_per_node] * draft_config.nnodes,
        use_gpu=True,
        max_colocate_count=3,
        name_prefix="decoupled_spec_draft_pool",
    )
    split_resource_pools = split_resource_pool(draft_resource_pool, draft_config.tp_size)
    if len(split_resource_pools) != draft_config.num_drafters:
        raise RuntimeError(
            f"Expected {draft_config.num_drafters} drafter resource pools, got {len(split_resource_pools)}"
        )

    draft_replicas = [
        SGLangReplica(
            replica_rank=replica_rank,
            config=draft_rollout_config,
            model_config=draft_model_config,
            gpus_per_node=draft_config.ngpus_per_node,
        )
        for replica_rank in range(draft_config.num_drafters)
    ]
    for replica in draft_replicas:
        replica.worker_name_prefix = "decoupled_draft_rollout"
        replica.server_name_prefix = "sglang_draft_server"
    return draft_replicas, split_resource_pools


def _check_draft_resources_available(draft_config: DraftConfig):
    available_resources = ray._private.state.available_resources_per_node()
    alive_node_ids = {node["NodeID"] for node in ray.nodes() if node.get("Alive")}
    eligible_nodes = []
    for node_id in alive_node_ids:
        node_resources = available_resources.get(node_id, {})
        available_gpus = int(node_resources.get("GPU", node_resources.get("NPU", 0)))
        if available_gpus >= draft_config.ngpus_per_node:
            eligible_nodes.append(node_id)

    if len(eligible_nodes) < draft_config.nnodes:
        raise ValueError(
            f"Need {draft_config.nnodes} free GPU nodes with at least "
            f"{draft_config.ngpus_per_node} GPUs each for decoupled-spec drafters, "
            f"but found {len(eligible_nodes)}"
        )


async def _setup_verifier_replicas(
    *,
    verifier_replicas: list[Any],
    rollout_config: RolloutConfig,
    worker_group: RayWorkerGroup | None,
    rollout_resource_pool: RayResourcePool | None,
):
    if worker_group:
        if rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    replica.setup_hybrid_colocated(worker_group, rollout_resource_pool)
                    for replica in verifier_replicas
                ]
            )
        else:
            await asyncio.gather(*[replica.setup_hybrid(worker_group) for replica in verifier_replicas])
    else:
        await asyncio.gather(*[replica.setup_standalone() for replica in verifier_replicas])


async def initialize_decoupled_spec_rollout_servers(
    *,
    config: DictConfig,
    rollout_config: RolloutConfig,
    model_config: HFModelConfig,
    rollout_replica_class: type,
    num_replicas: int,
    worker_group: RayWorkerGroup | None,
    rollout_resource_pool: RayResourcePool | None,
) -> tuple[list[Any], list[SGLangReplica]]:
    draft_config = get_draft_config(config)
    validate_decoupled_spec_config(rollout_config, draft_config)

    verifier_replicas = _create_verifier_replicas(
        rollout_replica_class=rollout_replica_class,
        rollout_config=rollout_config,
        model_config=model_config,
        num_replicas=num_replicas,
    )
    await _setup_verifier_replicas(
        verifier_replicas=verifier_replicas,
        rollout_config=rollout_config,
        worker_group=worker_group,
        rollout_resource_pool=rollout_resource_pool,
    )
    draft_replicas, draft_resource_pools = _create_drafter_replicas(
        rollout_config=rollout_config,
        model_config=model_config,
        draft_config=draft_config,
    )
    await asyncio.gather(
        *[
            replica.setup_standalone_with_resource_pool(resource_pool)
            for replica, resource_pool in zip(draft_replicas, draft_resource_pools, strict=True)
        ]
    )

    verifier_infos, drafter_infos = await asyncio.gather(
        asyncio.gather(*[replica.prepare_server_actors() for replica in verifier_replicas]),
        asyncio.gather(*[replica.prepare_server_actors() for replica in draft_replicas]),
    )
    topology = create_decoupled_spec_topology(verifier_infos, drafter_infos)

    verifier_server_configs = [
        endpoint_config.to_server_config(
            algorithm="DECOUPLED_VERIFY",
            speculative_num_steps=draft_config.speculative_num_steps,
            trace_dir=draft_config.trace_dir,
        )
        for endpoint_config in topology.verifier_configs
    ]
    drafter_server_configs = [
        endpoint_config.to_server_config(
            algorithm="DECOUPLED_DRAFT",
            speculative_num_steps=draft_config.speculative_num_steps,
            trace_dir=draft_config.trace_dir,
        )
        for endpoint_config in topology.drafter_configs
    ]
    for replica, server_config in zip(verifier_replicas, verifier_server_configs, strict=True):
        replica.set_decoupled_spec_config(server_config)
    for replica, server_config in zip(draft_replicas, drafter_server_configs, strict=True):
        replica.set_decoupled_spec_config(server_config)

    await asyncio.gather(
        *[replica.launch_prepared_servers() for replica in draft_replicas],
        *[replica.launch_prepared_servers() for replica in verifier_replicas],
    )

    return verifier_replicas, draft_replicas
