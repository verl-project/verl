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

import asyncio
import logging
import os
import threading
from collections import defaultdict
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.ray_utils import auto_await
from verl.utils.tracking import RLInsightLogger
from verl.workers.config import DistillationConfig, DistillationTeacherModelConfig
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import get_rollout_replica_class
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@auto_await
async def _run_all(tasks: list[asyncio.Task]):
    await asyncio.gather(*tasks)


class TeacherModelManager:
    """Teacher model manager."""

    def __init__(
        self,
        distillation_config: DistillationConfig,
        teacher_model_config: DistillationTeacherModelConfig,
        resource_pool: RayResourcePool,
    ):
        """
        Initialize the teacher model manager.

        Args:
            distillation_config (DistillationConfig): Distillation configuration.
            teacher_model_config (DistillationTeacherModelConfig): Teacher model configuration.
            resource_pool (RayResourcePool): Dedicated teacher resource pool.
        """

        # Need dataclass conversion for max_logprobs handling in post_init
        self.distillation_config = distillation_config
        self.teacher_model_config = teacher_model_config
        self.resource_pool = resource_pool
        self._initialize_llm_servers()
        if distillation_config.teacher_execution.mode == "dedicated":
            self._initialize_load_balancer_handle()

    def _initialize_llm_servers(self):
        teacher_model_config = self.teacher_model_config
        per_replica_world_size = teacher_model_config.per_replica_world_size
        num_replicas = teacher_model_config.num_replicas
        expected_pool_size = num_replicas * per_replica_world_size
        if self.resource_pool.world_size != expected_pool_size:
            raise ValueError(
                f"Teacher {teacher_model_config.key!r} expected sub-pool of size "
                f"{expected_pool_size} (num_replicas={num_replicas} * "
                f"per_replica_world_size={per_replica_world_size}), but got "
                f"{self.resource_pool.world_size}."
            )

        gpus_per_node = self.distillation_config.n_gpus_per_node
        rollout_replica_class = get_rollout_replica_class(teacher_model_config.inference.name)
        rollout_config = teacher_model_config.inference
        # Keep the model path unresolved until the rollout server actor is
        # placed. Each server backend converts this DictConfig to HFModelConfig
        # in its own process, so HDFS models are copied into that node's local
        # cache instead of the controller's local /tmp.
        model_config = OmegaConf.create(
            {
                "_target_": "verl.workers.config.HFModelConfig",
                "path": teacher_model_config.model_path,
            }
        )
        name_suffix = (teacher_model_config.key or "").replace("/", "_")
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=gpus_per_node,
                is_teacher_model=True,
                name_suffix=name_suffix,
            )
            for replica_rank in range(num_replicas)
        ]
        split_resource_pools = split_resource_pool(self.resource_pool, split_size=per_replica_world_size)
        assert len(split_resource_pools) == len(self.rollout_replicas)
        self._validate_replica_node_alignment(split_resource_pools, per_replica_world_size, gpus_per_node)
        _run_all(
            [
                server.init_colocated(resource_pool)
                for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
            ]
        )
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        needs_metrics = rollout_config.prometheus.enable or RLInsightLogger.enabled()
        if rollout_config.disable_log_stats and needs_metrics:
            raise ValueError("Metrics monitoring requires disable_log_stats=False, but it is currently True.")
        if not rollout_config.disable_log_stats and rollout_config.prometheus.enable:
            update_prometheus_config(rollout_config.prometheus, self.server_addresses, rollout_config.name)
        if not rollout_config.disable_log_stats and RLInsightLogger.enabled():
            RLInsightLogger.register_rollout_metrics(
                self.server_addresses,
                rollout_config.name,
                labels=[{"replica": f"teacher_{rank}"} for rank, server in enumerate(self.rollout_replicas)],
            )

    def _validate_replica_node_alignment(self, replica_pools, per_replica_world_size, gpus_per_node):
        """Verify that each replica occupies the expected number of nodes.

        `per_replica_world_size` (W below) is the GPU count of a *single* inference
        replica — the product of the replica's inference-time parallelism
        (tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size).
        It is not the teacher's total GPU footprint (`num_replicas * W`).

        `split_resource_pool` walks bundles linearly and is oblivious to node
        boundaries, so a replica's sub-pool can end up touching more nodes than W
        implies when W does not divide the node layout cleanly.

        Example (P = n_gpus_per_node = 4, two teachers with W=3 and W=4):

                node 0                  node 1
                [0 1 2 3]               [4 5 6 7]         ← bundle idx

            teacher A (W=3):
                [A A A .]               [. . . .]          expected span 1, observed 1  ✓
            teacher B (W=4):
                [. . . B]               [B B B .]          expected span 1, observed 2  ✗

        Teacher B's one replica (W=4) is expected to stay on a single node, but the
        linear split dropped it on bundles 3-6 — straddling nodes 0 and 1.
        """
        key = self.teacher_model_config.key
        P = gpus_per_node
        W = per_replica_world_size
        expected_span = (W + P - 1) // P
        for i, sub_pool in enumerate(replica_pools):
            start = sub_pool.start_bundle_index
            first_node = start // P
            last_node = (start + W - 1) // P
            observed_span = last_node - first_node + 1
            if observed_span != expected_span:
                raise ValueError(
                    f"Teacher {key!r} replica {i} sub-pool bundles [{start}, {start + W}) "
                    f"span {observed_span} node(s) but per_replica_world_size {W} with "
                    f"n_gpus_per_node {P} expects {expected_span}. Reorder teachers or "
                    f"adjust num_replicas / inference parallelism so each replica sub-pool "
                    f"aligns to node boundaries."
                )

    def _initialize_load_balancer_handle(self):
        from verl.workers.rollout.router import get_router_handle

        self.load_balancer_handle = get_router_handle(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            router_config_path=self.teacher_model_config.inference.router_config_path,
        )


def validate_hidden_state_checkpoints(config: DictConfig, teacher_models: dict, *, shared: bool = True) -> None:
    """Validate the token mapping for all teachers and the model layout for a weight-swapped pool."""
    from transformers import AutoConfig, AutoTokenizer

    from verl.trainer.distillation.hidden_state import teacher_output_scale
    from verl.utils.fs import copy_to_local

    actor = config.actor_rollout_ref.model
    trust_actor = actor.get("trust_remote_code", False)
    actor_cfg = AutoConfig.from_pretrained(
        copy_to_local(actor.get("hf_config_path") or actor.path), trust_remote_code=trust_actor
    )
    student_vocab_size = getattr(actor_cfg, "text_config", actor_cfg).vocab_size
    actor_tokenizer = AutoTokenizer.from_pretrained(
        copy_to_local(actor.get("tokenizer_path") or actor.path), trust_remote_code=trust_actor
    )
    student_vocab = actor_tokenizer.get_vocab()
    reference_signature = None
    for key, teacher in teacher_models.items():
        path = copy_to_local(teacher.model_path)
        cfg = AutoConfig.from_pretrained(path)
        text_config = getattr(cfg, "text_config", cfg)
        teacher_output_scale(text_config)
        if getattr(cfg, "quantization_config", None) or getattr(text_config, "quantization_config", None):
            raise ValueError(f"Teacher {key!r} requires an unquantized checkpoint for full-vocabulary KL.")
        teacher_vocab = AutoTokenizer.from_pretrained(path).get_vocab()
        if text_config.vocab_size != student_vocab_size or teacher_vocab != student_vocab:
            raise ValueError(f"Teacher {key!r} must have the student vocabulary size and token-to-id mapping.")
        if not shared:
            continue
        # The same instantiated model will execute all checkpoints. Include the whole architecture
        # (also vision/expert/rotary settings), excluding only provenance and fine-tuning metadata.
        signature = cfg.to_dict()
        for name in ("_name_or_path", "transformers_version", "finetuning_task", "id2label", "label2id"):
            signature.pop(name, None)
        if reference_signature is not None and signature != reference_signature:
            different = sorted(
                k
                for k in signature.keys() | reference_signature.keys()
                if signature.get(k) != reference_signature.get(k)
            )
            raise ValueError(f"Teacher {key!r} has incompatible vLLM model config fields {different}.")
        reference_signature = signature


class HiddenStateTeacherPool:
    """Route requests to their resident teachers and scatter returned handles into the input order."""

    def __init__(
        self, teacher_models, servers_by_key, *, weight_swapped, transfer_signal="hidden_states", chunk_size=32
    ):
        self.teacher_models = teacher_models
        self.servers_by_key = servers_by_key
        self.weight_swapped = weight_swapped
        self.transfer_signal = transfer_signal
        self.chunk_size = chunk_size
        self.teacher_indices = {key: index for index, key in enumerate(teacher_models)}
        self.resident_key = next(iter(teacher_models))
        self.weight_version = 1
        self._lock = threading.Lock()

    def _switch_to(self, key):
        import ray

        if self.resident_key == key:
            return
        # Any failed reload can leave different replica ranks on different weights.
        # Force a complete reload of the next group, including when retrying the old teacher.
        self.resident_key = None
        refs = []
        try:
            for server in self.servers_by_key[key]:
                refs.append(server.switch_teacher_weights.remote(self.teacher_models[key].model_path))
        finally:
            if refs:
                ray.wait(refs, num_returns=len(refs), fetch_local=False)
        ray.get(refs)
        self.resident_key = key
        self.weight_version += 1
        logger.info("Shared teacher pool switched to key=%s version=%s", key, self.weight_version)

    def compute(self, requests):
        groups = defaultdict(list)
        single_key = next(iter(self.teacher_models)) if len(self.teacher_models) == 1 else None
        for index, request in enumerate(requests):
            key = single_key if single_key is not None else request.get("routing_key")
            if key not in self.teacher_models:
                raise ValueError(f"Unknown teacher routing key {key!r}. Configured keys: {list(self.teacher_models)}.")
            groups[key].append((index, request))

        handles = [None] * len(requests)
        with self._lock:
            if self.weight_swapped:
                # Drain the currently resident group before switching every replica to the next checkpoint.
                order = sorted(groups, key=lambda key: (key != self.resident_key, self.teacher_indices[key]))
                for key in order:
                    self._switch_to(key)
                    self._compute_groups({key: groups[key]}, handles)
            else:
                # Dedicated pools are independent. Submit every teacher group before waiting.
                self._compute_groups(groups, handles)
        return handles

    def _compute_groups(self, groups, handles):
        import ray

        refs = []
        expected_handles = []
        try:
            for key, group in groups.items():
                servers = self.servers_by_key[key]
                for offset, (index, request) in enumerate(group):
                    queue_request = request.copy()
                    queue_request.pop("routing_key", None)
                    mm_data = queue_request.pop("multi_modal_data", None)
                    mm_kwargs = queue_request.pop("mm_processor_kwargs", None)
                    queue_request.update(
                        teacher_key=key,
                        teacher_index=self.teacher_indices[key],
                        teacher_weight_version=self.weight_version,
                    )
                    if self.transfer_signal == "full_logits":
                        queue_request.update(
                            signal_kind="full_logits",
                            teacher_model_path=self.teacher_models[key].model_path,
                            projection_chunk_size=self.chunk_size,
                        )
                    server = servers[offset % len(servers)]
                    refs.append(
                        server.compute_teacher_hidden_states.remote(
                            request_id=uuid4().hex,
                            prompt_ids=queue_request["sequence_ids"],
                            queue_request=queue_request,
                            multi_modal_data=mm_data,
                            mm_processor_kwargs=mm_kwargs,
                        )
                    )
                    expected_handles.append((index, queue_request))
        finally:
            # A producer may fail before its peers. All producers must finish before the trainer clears the step.
            if refs:
                ray.wait(refs, num_returns=len(refs), fetch_local=False)
        identity = ("uid", "batch_index", "teacher_key", "teacher_index", "teacher_weight_version", "sequence_hash")
        for (index, request), handle in zip(expected_handles, ray.get(refs), strict=True):
            if any(handle[name] != request[name] for name in identity):
                raise ValueError(f"Incorrect teacher hidden-state handle for index {request['batch_index']}.")
            handles[index] = handle


class MultiTeacherModelManager:
    """Manages one inner `TeacherModelManager` per teacher model, keyed by each teacher's `key`."""

    def __init__(
        self,
        config: DictConfig,
        resource_pool: RayResourcePool,
    ):
        """
        Initialize the multi-teacher model manager.

        Args:
            config (DictConfig): Full configuration.
            resource_pool (RayResourcePool): Combined resource pool for all teachers.
        """
        self.config = config
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)

        self.resource_pool = resource_pool
        self.teacher_model_managers: dict[str, TeacherModelManager] = {}
        self.server_addresses: dict[str, list[str]] = {}
        self.server_handles: dict[str, list] = {}
        self.load_balancer_handle: dict[str, object] = {}

        self._initialize_teacher_model_managers()

    def _initialize_teacher_model_managers(self):
        teacher_models = self.distillation_config.teacher_models
        mode = self.distillation_config.teacher_execution.mode
        self.hidden_pool = None
        if mode != "dedicated":
            actor = self.config.actor_rollout_ref.actor
            if actor.strategy not in ("fsdp", "fsdp2", "megatron"):
                raise ValueError(f"Hidden-state OPD requires an FSDP or Megatron actor, got {actor.strategy!r}.")
            if actor.get("use_fused_kernels", False) or self.config.actor_rollout_ref.model.get(
                "use_fused_kernels", False
            ):
                raise ValueError("Hidden-state OPD requires use_fused_kernels=False to access student logits.")
            storage = OmegaConf.select(self.config, "transfer_queue.backend.storage_backend", default="SimpleStorage")
            if storage != "SimpleStorage":
                raise ValueError("Hidden-state OPD requires transfer_queue.backend.storage_backend=SimpleStorage.")
            validate_hidden_state_checkpoints(self.config, teacher_models, shared=mode == "weight_swapped_hidden_state")
        if mode == "weight_swapped_hidden_state":
            first = next(iter(teacher_models.values()))
            manager = TeacherModelManager(self.distillation_config, first, self.resource_pool)
            self.teacher_model_managers[first.key] = manager
            self.hidden_pool = HiddenStateTeacherPool(
                teacher_models, dict.fromkeys(teacher_models, manager.server_handles), weight_swapped=True
            )
            return
        split_sizes = [teacher.world_size for teacher in teacher_models.values()]
        split_pools = split_resource_pool(self.resource_pool, split_size=split_sizes)

        for (key, teacher_model_config), teacher_pool in zip(teacher_models.items(), split_pools, strict=True):
            manager = TeacherModelManager(
                distillation_config=self.distillation_config,
                teacher_model_config=teacher_model_config,
                resource_pool=teacher_pool,
            )
            self.teacher_model_managers[key] = manager
            self.server_addresses[key] = manager.server_addresses
            self.server_handles[key] = manager.server_handles
            if mode == "dedicated":
                self.load_balancer_handle[key] = manager.load_balancer_handle
        if mode == "dedicated_hidden_state":
            self.hidden_pool = HiddenStateTeacherPool(
                teacher_models,
                self.server_handles,
                weight_swapped=False,
                transfer_signal=self.distillation_config.teacher_execution.transfer_signal,
                chunk_size=self.distillation_config.distillation_loss.hidden_state_chunk_size,
            )

    def compute_hidden_states(self, requests: list[dict]) -> list[dict]:
        if self.hidden_pool is None:
            raise ValueError("This teacher execution mode cannot produce hidden states.")
        return self.hidden_pool.compute(requests)

    def get_client(self) -> dict[str, LLMServerClient] | None:
        """Get the LLMServerClient for each teacher model."""
        if self.hidden_pool is not None:
            return None
        teacher_clients = {}
        for key, manager in self.teacher_model_managers.items():
            teacher_clients[key] = LLMServerClient(
                config=self.config,
                load_balancer_handle=manager.load_balancer_handle,
            )
        return teacher_clients
