from __future__ import annotations

import asyncio
from dataclasses import dataclass

import ray

from verl.single_controller.ray import RayWorkerGroup
from verl.workers.rollout.replica import RolloutReplica

from .layout import (
    build_draft_model_config,
    build_draft_rollout_config,
    compute_decoupled_spec_topology,
)
from .replica import DraftSGLangReplica, VerifySGLangReplica


@dataclass
class DecoupledSpecServerInitResult:
    rollout_replicas: list[RolloutReplica]
    server_handles: list[ray.actor.ActorHandle]
    server_addresses: list[str]


async def initialize_decoupled_spec_llm_servers(
    *,
    rollout_config,
    model_config,
    worker_group: RayWorkerGroup,
) -> DecoupledSpecServerInitResult:
    if worker_group is None:
        raise NotImplementedError("decoupled speculation currently only supports hybrid_engine rollout")

    topo = compute_decoupled_spec_topology(rollout_config, world_size=worker_group.world_size)
    draft_rollout_config = build_draft_rollout_config(rollout_config)
    draft_model_config = build_draft_model_config(model_config, rollout_config.draft.model_path)

    draft_replicas = [
        DraftSGLangReplica(
            replica_rank=replica_rank,
            config=draft_rollout_config,
            model_config=draft_model_config,
            gpus_per_node=rollout_config.n_gpus_per_node,
            topology_rollout_config=rollout_config,
        )
        for replica_rank in range(topo.num_draft_replicas)
    ]
    await asyncio.gather(*[replica.init_hybrid_decoupled(worker_group) for replica in draft_replicas])
    draft_actor_names = [replica.draft_actor_name for replica in draft_replicas]

    verify_replicas = [
        VerifySGLangReplica(
            replica_rank=topo.get_shared_replica_rank("verify", verify_replica_rank),
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=rollout_config.n_gpus_per_node,
            draft_actor_names=draft_actor_names,
        )
        for verify_replica_rank in range(topo.num_verify_replicas)
    ]
    await asyncio.gather(*[replica.init_hybrid_decoupled(worker_group) for replica in verify_replicas])

    rollout_replicas: list[RolloutReplica] = [*draft_replicas, *verify_replicas]
    return DecoupledSpecServerInitResult(
        rollout_replicas=rollout_replicas,
        server_handles=[replica._server_handle for replica in verify_replicas],
        server_addresses=[replica._server_address for replica in verify_replicas],
    )
