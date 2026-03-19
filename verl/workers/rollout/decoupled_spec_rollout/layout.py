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
class ReplicaAssignment:
    role: DecoupledSpecRole
    replica_rank: int
    world_size: int
    worker_indices: tuple[int, ...]

    @property
    def start_idx(self) -> int:
        return self.worker_indices[0]

    @property
    def end_idx(self) -> int:
        return self.worker_indices[-1] + 1

    def contains_rank(self, global_rank: int) -> bool:
        return self.start_idx <= global_rank < self.end_idx

    def offset_for_rank(self, global_rank: int) -> int:
        if not self.contains_rank(global_rank):
            raise ValueError(f"Rank {global_rank} does not belong to assignment {self}")
        return global_rank - self.start_idx


@dataclass(frozen=True)
class DecoupledSpecLayout:
    world_size: int
    verify_world_size: int
    draft_world_size: int
    verify_assignments: tuple[ReplicaAssignment, ...]
    draft_assignments: tuple[ReplicaAssignment, ...]

    @property
    def verify_gpu_count(self) -> int:
        return len(self.verify_assignments) * self.verify_world_size

    @property
    def draft_gpu_count(self) -> int:
        return len(self.draft_assignments) * self.draft_world_size

    @property
    def num_verify_replicas(self) -> int:
        return len(self.verify_assignments)

    @property
    def num_draft_replicas(self) -> int:
        return len(self.draft_assignments)

    def all_assignments(self) -> tuple[ReplicaAssignment, ...]:
        return self.verify_assignments + self.draft_assignments

    def assignment_for_rank(self, global_rank: int) -> ReplicaAssignment:
        for assignment in self.all_assignments():
            if assignment.contains_rank(global_rank):
                return assignment
        raise ValueError(f"Global rank {global_rank} is outside of decoupled spec layout")


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


def build_decoupled_spec_layout(
    config: RolloutConfig | DictConfig | dict,
    world_size: int,
) -> DecoupledSpecLayout:
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

    verify_assignments = []
    for replica_rank in range(num_verify_replicas):
        start_idx = replica_rank * verify_world_size
        verify_assignments.append(
            ReplicaAssignment(
                role=DecoupledSpecRole.VERIFY,
                replica_rank=replica_rank,
                world_size=verify_world_size,
                worker_indices=tuple(range(start_idx, start_idx + verify_world_size)),
            )
        )

    draft_assignments = []
    draft_offset = verify_gpu_count
    for replica_rank in range(num_draft_replicas):
        start_idx = draft_offset + replica_rank * draft_world_size
        draft_assignments.append(
            ReplicaAssignment(
                role=DecoupledSpecRole.DRAFT,
                replica_rank=replica_rank,
                world_size=draft_world_size,
                worker_indices=tuple(range(start_idx, start_idx + draft_world_size)),
            )
        )

    return DecoupledSpecLayout(
        world_size=world_size,
        verify_world_size=verify_world_size,
        draft_world_size=draft_world_size,
        verify_assignments=tuple(verify_assignments),
        draft_assignments=tuple(draft_assignments),
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

    layout = build_decoupled_spec_layout(rollout_config, world_size=world_size)
    assignment = layout.assignment_for_rank(global_rank)
    rollout_rank = assignment.offset_for_rank(global_rank)
    node_rank = rollout_rank // local_world_size
    local_rank = rollout_rank % local_world_size
    server_prefix = "sglang_server" if assignment.role == DecoupledSpecRole.VERIFY else "sglang_draft_server"
    return ServerAdapterLayout(
        role=assignment.role,
        replica_rank=assignment.replica_rank,
        rollout_rank=rollout_rank,
        node_rank=node_rank,
        local_rank=local_rank,
        server_actor_name=f"{server_prefix}_{assignment.replica_rank}_{node_rank}",
    )
