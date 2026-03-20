from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig


class DecoupledSpecRole(str, Enum):
    VERIFY = "verify"
    DRAFT = "draft"


@dataclass(frozen=True)
class DecoupledSpecTopology:
    """GPU / replica counts for decoupled speculative rollout (verify first, then draft in global rank space)."""

    world_size: int
    verify_world_size: int
    draft_world_size: int
    verify_gpu_count: int
    draft_gpu_count: int
    num_verify_replicas: int
    num_draft_replicas: int


@dataclass(frozen=True)
class ServerAdapterLayout:
    role: DecoupledSpecRole
    replica_rank: int
    rollout_rank: int
    node_rank: int
    local_rank: int
    server_actor_name: str


def _clone_config(config: RolloutConfig | DictConfig | dict) -> RolloutConfig | DictConfig:
    if isinstance(config, DictConfig | dict):
        return OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    return copy.deepcopy(config)


def build_draft_rollout_config(config: RolloutConfig | DictConfig | dict) -> RolloutConfig | DictConfig:
    draft_config = _clone_config(config)
    if isinstance(draft_config, DictConfig):
        draft_cfg = draft_config.draft
        draft_config.tensor_model_parallel_size = draft_cfg.tensor_model_parallel_size
        draft_config.data_parallel_size = 1
        draft_config.expert_parallel_size = 1
        draft_config.pipeline_model_parallel_size = 1
        if "mtp" in draft_config and "enable_rollout" in draft_config.mtp:
            draft_config.mtp.enable_rollout = False
        return draft_config

    draft_cfg = draft_config.draft
    draft_config.tensor_model_parallel_size = draft_cfg.tensor_model_parallel_size
    draft_config.data_parallel_size = 1
    draft_config.expert_parallel_size = 1
    draft_config.pipeline_model_parallel_size = 1
    if draft_config.mtp is not None:
        draft_config.mtp.enable_rollout = False
    return draft_config


def build_draft_model_config(model_config: DictConfig | dict, draft_model_path: str) -> DictConfig:
    draft_model_config = OmegaConf.create(OmegaConf.to_container(model_config, resolve=False))
    draft_model_config.path = draft_model_path
    draft_model_config.local_path = None
    draft_model_config.hf_config_path = draft_model_path
    draft_model_config.local_hf_config_path = None
    draft_model_config.tokenizer_path = draft_model_path
    draft_model_config.local_tokenizer_path = None
    draft_model_config.hf_config = None
    draft_model_config.generation_config = None
    draft_model_config.tokenizer = None
    draft_model_config.processor = None
    draft_model_config.architectures = None
    return draft_model_config


def compute_decoupled_spec_topology(
    config: RolloutConfig | DictConfig | dict,
    world_size: int,
) -> DecoupledSpecTopology:
    """Compute verify/draft GPU counts and replica counts for decoupled speculative rollout.

    Global worker layout (indices in ``worker_group.workers``):
    - ranks ``[0, verify_gpu_count)``: verify replicas (each replica is ``verify_world_size`` workers)
    - ranks ``[verify_gpu_count, world_size)``: draft replicas (each replica is ``draft_world_size`` workers)
    """
    rollout_config = omega_conf_to_dataclass(config, dataclass_type=RolloutConfig)
    verify_world_size = (
        rollout_config.tensor_model_parallel_size
        * rollout_config.data_parallel_size
        * rollout_config.pipeline_model_parallel_size
    )
    draft_world_size = rollout_config.draft.tensor_model_parallel_size
    draft_gpu_count = rollout_config.draft.ngpus

    if draft_world_size <= 0:
        raise ValueError("draft.tensor_model_parallel_size must be > 0")
    if draft_gpu_count % draft_world_size != 0:
        raise ValueError("draft.ngpus must be divisible by draft.tensor_model_parallel_size")
    if world_size <= verify_world_size + draft_gpu_count:
        raise ValueError(
            "world_size must be greater than rollout tp*dp*pp plus draft.ngpus "
            f"(got world_size={world_size}, verify_world_size={verify_world_size}, draft_ngpus={draft_gpu_count})"
        )

    verify_gpu_count = world_size - draft_gpu_count
    if verify_gpu_count % verify_world_size != 0:
        raise ValueError(
            "The remaining verify GPUs must be divisible by rollout tp*dp*pp "
            f"(got verify_gpu_count={verify_gpu_count}, verify_world_size={verify_world_size})"
        )

    num_verify_replicas = verify_gpu_count // verify_world_size
    num_draft_replicas = draft_gpu_count // draft_world_size

    return DecoupledSpecTopology(
        world_size=world_size,
        verify_world_size=verify_world_size,
        draft_world_size=draft_world_size,
        verify_gpu_count=verify_gpu_count,
        draft_gpu_count=draft_gpu_count,
        num_verify_replicas=num_verify_replicas,
        num_draft_replicas=num_draft_replicas,
    )


def resolve_server_adapter_layout(
    config: RolloutConfig | DictConfig | dict,
    global_rank: int,
    local_world_size: int,
    world_size: int,
) -> Optional[ServerAdapterLayout]:
    rollout_config = omega_conf_to_dataclass(config, dataclass_type=RolloutConfig)
    if not rollout_config.enable_decoupled_spec:
        return None

    topo = compute_decoupled_spec_topology(rollout_config, world_size=world_size)

    if global_rank < topo.verify_gpu_count:
        replica_rank = global_rank // topo.verify_world_size
        rollout_rank = global_rank - replica_rank * topo.verify_world_size
        role = DecoupledSpecRole.VERIFY
        server_prefix = "sglang_server"
    else:
        offset = global_rank - topo.verify_gpu_count
        replica_rank = offset // topo.draft_world_size
        rollout_rank = offset - replica_rank * topo.draft_world_size
        role = DecoupledSpecRole.DRAFT
        server_prefix = "sglang_draft_server"

    node_rank = rollout_rank // local_world_size
    local_rank = rollout_rank % local_world_size
    return ServerAdapterLayout(
        role=role,
        replica_rank=replica_rank,
        rollout_rank=rollout_rank,
        node_rank=node_rank,
        local_rank=local_rank,
        server_actor_name=f"{server_prefix}_{replica_rank}_{node_rank}",
    )
