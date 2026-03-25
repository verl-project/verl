from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig


class DecoupledSpecRole(str, Enum):
    VERIFY = "verify"
    DRAFT = "draft"


@dataclass(frozen=True)
class DecoupledSpecTopology:
    """GPU / replica counts for decoupled speculative rollout.

    Worker/GPU layout is draft-first in global rank space:
    - ranks ``[0, draft_gpu_count)`` belong to draft replicas
    - ranks ``[draft_gpu_count, draft_gpu_count + verify_gpu_count)`` belong to verifier replicas

    Public replica ranks are shared across roles and ordered draft-first so they
    can be reasoned about as a single namespace:
    - drafts use shared replica ranks ``[0, num_draft_replicas)``
    - verifiers use shared replica ranks
      ``[num_draft_replicas, num_draft_replicas + num_verify_replicas)``
    """

    world_size: int
    verify_world_size: int
    draft_world_size: int
    verify_gpu_count: int
    draft_gpu_count: int
    num_verify_replicas: int
    num_draft_replicas: int

    @property
    def num_total_replicas(self) -> int:
        return self.num_draft_replicas + self.num_verify_replicas

    @property
    def verify_replica_rank_start(self) -> int:
        return self.num_draft_replicas

    @property
    def verify_replica_rank_end(self) -> int:
        return self.verify_replica_rank_start + self.num_verify_replicas

    @property
    def assigned_gpu_count(self) -> int:
        return self.draft_gpu_count + self.verify_gpu_count

    def get_shared_replica_rank(self, role: DecoupledSpecRole | str, role_replica_rank: int) -> int:
        role = DecoupledSpecRole(role)
        if role == DecoupledSpecRole.DRAFT:
            if not 0 <= role_replica_rank < self.num_draft_replicas:
                raise ValueError(
                    f"draft role_replica_rank out of range: {role_replica_rank}, "
                    f"num_draft_replicas={self.num_draft_replicas}"
                )
            return role_replica_rank

        if not 0 <= role_replica_rank < self.num_verify_replicas:
            raise ValueError(
                f"verify role_replica_rank out of range: {role_replica_rank}, "
                f"num_verify_replicas={self.num_verify_replicas}"
            )
        return self.verify_replica_rank_start + role_replica_rank

    def get_role_replica_rank(self, role: DecoupledSpecRole | str, shared_replica_rank: int) -> int:
        role = DecoupledSpecRole(role)
        if role == DecoupledSpecRole.DRAFT:
            if not 0 <= shared_replica_rank < self.num_draft_replicas:
                raise ValueError(
                    f"shared draft replica_rank out of range: {shared_replica_rank}, "
                    f"num_draft_replicas={self.num_draft_replicas}"
                )
            return shared_replica_rank

        verify_start = self.verify_replica_rank_start
        verify_end = self.verify_replica_rank_end
        if not verify_start <= shared_replica_rank < verify_end:
            raise ValueError(
                f"shared verify replica_rank out of range: {shared_replica_rank}, "
                f"verify_rank_range=[{verify_start}, {verify_end})"
            )
        return shared_replica_rank - verify_start

    def get_replica_worker_range(self, role: DecoupledSpecRole | str, shared_replica_rank: int) -> tuple[int, int]:
        role = DecoupledSpecRole(role)
        role_replica_rank = self.get_role_replica_rank(role, shared_replica_rank)
        if role == DecoupledSpecRole.DRAFT:
            start = role_replica_rank * self.draft_world_size
            end = start + self.draft_world_size
        else:
            start = self.draft_gpu_count + role_replica_rank * self.verify_world_size
            end = start + self.verify_world_size
        return start, end

    def get_replica_base_gpu_id(
        self,
        role: DecoupledSpecRole | str,
        shared_replica_rank: int,
        gpus_per_node: int,
    ) -> int:
        if gpus_per_node <= 0:
            raise ValueError(f"gpus_per_node must be > 0, got {gpus_per_node}")
        start, _ = self.get_replica_worker_range(role, shared_replica_rank)
        return start % gpus_per_node


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
    - ranks ``[0, draft_gpu_count)``: draft replicas (each replica is ``draft_world_size`` workers)
    - ranks ``[draft_gpu_count, draft_gpu_count + verify_gpu_count)``:
      verify replicas (each replica is ``verify_world_size`` workers)
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
    if world_size < verify_world_size + draft_gpu_count:
        raise ValueError(
            "world_size must be greater than rollout tp*dp*pp plus draft.ngpus "
            f"(got world_size={world_size}, verify_world_size={verify_world_size}, draft_ngpus={draft_gpu_count})"
        )

    verify_gpu_count = world_size - draft_gpu_count
    verify_gpu_count -= verify_gpu_count % verify_world_size # 保证 verify_gpu_count 是 verify_world_size 的倍数

    if verify_gpu_count % verify_world_size != 0:
        raise ValueError(
            "The remaining verify GPUs must be divisible by rollout tp*dp*pp "
            f"(got verify_gpu_count={verify_gpu_count}, verify_world_size={verify_world_size})"
        )

    num_verify_replicas = verify_gpu_count // verify_world_size
    num_draft_replicas = draft_gpu_count // draft_world_size
    print(
        "[decoupled_spec][topology] compute_topology "
        f"world_size={world_size} verify_world_size={verify_world_size} "
        f"draft_world_size={draft_world_size} verify_gpu_count={verify_gpu_count} "
        f"draft_gpu_count={draft_gpu_count} num_verify_replicas={num_verify_replicas} "
        f"num_draft_replicas={num_draft_replicas}"
    )

    return DecoupledSpecTopology(
        world_size=world_size,
        verify_world_size=verify_world_size,
        draft_world_size=draft_world_size,
        verify_gpu_count=verify_gpu_count,
        draft_gpu_count=draft_gpu_count,
        num_verify_replicas=num_verify_replicas,
        num_draft_replicas=num_draft_replicas,
    )


def build_rollout_device_mesh(
    config: RolloutConfig | DictConfig | dict,
    global_rank: int,
    world_size: int,
    device_type: str,
) -> Optional[DeviceMesh]:
    rollout_config = omega_conf_to_dataclass(config, dataclass_type=RolloutConfig)
    infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
    infer_pp = rollout_config.pipeline_model_parallel_size

    topo = compute_decoupled_spec_topology(rollout_config, world_size=world_size)

    if global_rank < topo.draft_gpu_count:
        # Draft workers do not participate in verifier-side weight sync today.
        return None
    elif global_rank < topo.assigned_gpu_count:
        verify_ranks = torch.arange(topo.draft_gpu_count, topo.assigned_gpu_count, dtype=torch.int)
        verify_mesh = verify_ranks.view(topo.num_verify_replicas, infer_tp, infer_pp)
        return DeviceMesh(device_type, mesh=verify_mesh, mesh_dim_names=("dp", "infer_tp", "infer_pp"))
    else:
        return None


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

    if global_rank < topo.draft_gpu_count:
        offset = global_rank
        role_replica_rank = offset // topo.draft_world_size
        replica_rank = topo.get_shared_replica_rank(DecoupledSpecRole.DRAFT, role_replica_rank)
        rollout_rank = offset - role_replica_rank * topo.draft_world_size
        role = DecoupledSpecRole.DRAFT
        server_prefix = "sglang_draft_server"
    elif global_rank < topo.assigned_gpu_count:
        offset = global_rank - topo.draft_gpu_count
        role_replica_rank = offset // topo.verify_world_size
        replica_rank = topo.get_shared_replica_rank(DecoupledSpecRole.VERIFY, role_replica_rank)
        rollout_rank = offset - role_replica_rank * topo.verify_world_size
        role = DecoupledSpecRole.VERIFY
        server_prefix = "sglang_server"
    else:
        return None

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
