# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Trainer-side P2P weight updater following the design of Miles ``UpdateWeightP2P``."""

from __future__ import annotations

import gc
import logging
from collections.abc import Generator, Mapping
from typing import Any

import torch

from .transfer_utils import (
    P2PRolloutTopology,
    P2PTransferManager,
    RemoteTransferPlan,
    RemoteWeightInfo,
    TransferTaskP2PMeta,
    create_server_args_from_dict,
    create_transfer_engine,
    iter_named_tensor_buckets,
    register_cpu_memory,
    unregister_cpu_memory,
)

logger = logging.getLogger(__name__)


class P2PTrainerWeightUpdater:
    """Miles-style trainer push: CPU pinned replica + Mooncake RDMA writes."""

    def __init__(
        self,
        *,
        model_path: str,
        rollout_topology: P2PRolloutTopology,
        num_workers: int = 4,
        transfer_timeout: float = 30.0,
        is_master: bool = False,
        bucket_size_bytes: int = 2 << 30,
    ) -> None:
        self.model_path = model_path
        self.is_master = is_master
        self._bucket_size_bytes = bucket_size_bytes
        self.transfer_plan = RemoteTransferPlan(rollout_topology)
        self.transfer_manager = P2PTransferManager(num_workers=num_workers, transfer_timeout=transfer_timeout)

        self._connected = False
        self._model_registered = False
        self._tensor_update_pending: dict[str, int] = {}
        self._staged_tensors: dict[str, list[tuple[str, torch.Tensor]]] = {}
        self._shared_params_dict: dict[str, torch.Tensor] = {}
        self._shared_param_mapper = None
        self._weight_memory_registry: dict[str, tuple[int, int, int]] = {}
        self._transfer_engine = None
        self._transfer_engine_meta_list: list[tuple[torch.nn.Module, list[RemoteWeightInfo]]] = []
        self._rollout_metadata: dict[str, Any] | None = None

    @property
    def is_source(self) -> bool:
        return self.transfer_plan.is_source

    def connect_rollout_metadata(self, rollout_metadata: dict[str, Any]) -> None:
        """Initialize transfer engine state from manager-collected rollout metadata."""
        self._rollout_metadata = rollout_metadata
        remote_weight_infos_by_session_id = rollout_metadata["remote_weight_infos_by_session_id"]
        targets_to_session_id = rollout_metadata["targets_to_session_id"]
        session_id_to_server_args = rollout_metadata["session_id_to_server_args"]
        if self._connected:
            return
        if not self.is_source:
            self._connected = True
            return

        targets = self.transfer_plan.plan_p2p()
        targets_grouped_by_engine_rank: dict[int, list[TransferTaskP2PMeta]] = {}
        for target in targets:
            targets_grouped_by_engine_rank.setdefault(target.engine_rank, []).append(target)

        self._transfer_engine = create_transfer_engine()
        first_engine_rank = True
        for rank_targets in targets_grouped_by_engine_rank.values():
            first_target = rank_targets[0]
            session_id = targets_to_session_id[(first_target.engine_ind, first_target.engine_rank)]
            parallelism_config = self._import_parallelism_config(remote_weight_infos_by_session_id[session_id][1])
            server_args = create_server_args_from_dict(session_id_to_server_args[session_id])

            model_replica = self._create_cpu_replica(
                parallelism_config,
                self.model_path,
                server_args,
                first_engine_rank=first_engine_rank,
            )
            if first_engine_rank:
                self._shared_params_dict = dict(model_replica.named_parameters())
                from sglang.srt.model_loader.parameter_mapper import ParameterMapper

                self._shared_param_mapper = ParameterMapper.from_model(model_replica)
                first_engine_rank = False

            remote_infos = []
            for target in rank_targets:
                session_id = targets_to_session_id[(target.engine_ind, target.engine_rank)]
                weights_info = remote_weight_infos_by_session_id[session_id][0]
                remote_infos.append(RemoteWeightInfo(session_id, weights_info))
            self._transfer_engine_meta_list.append((model_replica, remote_infos))

        self._connected = True
        if self.is_master:
            logger.info(
                "P2P trainer connected: gathered_dp_rank=%s targets=%s",
                self.transfer_plan.gathered_dp_rank,
                len(targets),
            )

    def release_for_checkpoint(self) -> None:
        """Free P2P CPU replica and Mooncake registrations before Megatron checkpoint save."""
        if not self._connected:
            return

        if self.is_source:
            self.transfer_manager.wait_transfers()
            unregister_cpu_memory(self._weight_memory_registry, self._transfer_engine)
            self._transfer_engine_meta_list.clear()
            self._shared_params_dict.clear()
            self._shared_param_mapper = None
            self._staged_tensors.clear()
            self._tensor_update_pending.clear()
            # Keep rollout metadata for restore_after_checkpoint (small, stable ptrs).
            self._transfer_engine = None

        self._model_registered = False
        self._connected = False
        gc.collect()
        if self.is_master:
            print("[P2PTrainerWeightUpdater] released CPU replica for checkpoint save", flush=True)

    def restore_after_checkpoint(self) -> None:
        """Recreate P2P CPU replica after checkpoint save."""
        if self._rollout_metadata is None or self._connected:
            return
        self.connect_rollout_metadata(self._rollout_metadata)
        if self.is_master:
            print("[P2PTrainerWeightUpdater] restored CPU replica after checkpoint save", flush=True)

    def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]) -> None:
        """Stage HF tensors, load CPU replica, and issue RDMA writes."""
        if not self._connected:
            raise RuntimeError("P2P trainer updater is not connected; call connect_rollout_metadata first")

        if not self.is_source:
            raise RuntimeError(
                "P2P send_weights must not be called on non-source trainer ranks; skip export/update_weights instead."
            )

        if not self._model_registered:
            self._weight_memory_registry = register_cpu_memory(self._shared_params_dict, self._transfer_engine)
            self._model_registered = True

        rdma_writes = 0
        num_tensors = 0
        total_bytes = 0
        for bucket in iter_named_tensor_buckets(weights, self._bucket_size_bytes):
            num_tensors += len(bucket)
            total_bytes += sum(tensor.numel() * tensor.element_size() for _, tensor in bucket)
            transfer_ready_params, ready_hf_tensors = self._get_transfer_ready_params(bucket)
            if transfer_ready_params and ready_hf_tensors:
                rdma_writes += self._transfer_ready_bucket(transfer_ready_params, ready_hf_tensors)

        self.transfer_manager.wait_transfers()
        assert not self._tensor_update_pending and not self._staged_tensors, (
            f"Pending staged tensors after P2P update: {self._tensor_update_pending}, {self._staged_tensors}"
        )

        if self.is_master:
            logger.info(
                "P2P send_weights: %s tensors, %.2f MiB, %s RDMA sessions",
                num_tensors,
                total_bytes / (1024 * 1024),
                rdma_writes,
            )

    def _get_transfer_ready_params(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]]
    ) -> tuple[list[str], list[tuple[str, torch.Tensor]]]:
        transfer_ready_params: list[str] = []
        params_dict = self._shared_params_dict
        assert self._shared_param_mapper is not None

        for name, tensor in converted_named_tensors:
            mapped_result = self._shared_param_mapper.map(name)
            mapped, num_shards, num_experts = (
                mapped_result.sglang_name,
                mapped_result.num_shards,
                mapped_result.num_local_experts,
            )
            if mapped not in params_dict:
                raise RuntimeError(
                    f"P2P mapped parameter {mapped!r} (from HF name {name!r}) "
                    "not found in shared CPU replica; skipping would leave rollout "
                    "weights stale"
                )

            if num_experts is not None and num_experts > 0:
                total_expected = num_experts * num_shards
            else:
                total_expected = num_shards

            self._staged_tensors.setdefault(mapped, []).append((name, tensor))

            if total_expected == 1:
                transfer_ready_params.append(mapped)
            else:
                if mapped not in self._tensor_update_pending:
                    self._tensor_update_pending[mapped] = total_expected - 1
                else:
                    self._tensor_update_pending[mapped] -= 1
                if self._tensor_update_pending[mapped] == 0:
                    transfer_ready_params.append(mapped)

        ready_hf_tensors: list[tuple[str, torch.Tensor]] = []
        for param_name in transfer_ready_params:
            staged = self._staged_tensors.pop(param_name, [])
            ready_hf_tensors.extend(staged)
            self._tensor_update_pending.pop(param_name, None)

        return transfer_ready_params, ready_hf_tensors

    def _transfer_ready_bucket(
        self, transfer_ready_params: list[str], ready_hf_tensors: list[tuple[str, torch.Tensor]]
    ) -> int:
        """Load one ready bucket into the CPU replica and issue RDMA writes."""
        rdma_writes = 0
        last_idx = len(self._transfer_engine_meta_list) - 1
        for i, (model_replica, remote_weight_infos) in enumerate(self._transfer_engine_meta_list):
            model_replica.load_weights(ready_hf_tensors)
            is_last = i == last_idx
            if is_last:
                for remote_session in remote_weight_infos:
                    self.transfer_manager.submit(
                        self._do_p2p_write_one_session,
                        remote_session,
                        transfer_ready_params,
                    )
                    rdma_writes += 1
            else:
                futures = [
                    self.transfer_manager.submit_returning_future(
                        self._do_p2p_write_one_session,
                        remote_session,
                        transfer_ready_params,
                    )
                    for remote_session in remote_weight_infos
                ]
                self.transfer_manager.wait_futures(futures)
                rdma_writes += len(futures)
        return rdma_writes

    def _do_p2p_write_one_session(self, remote_session: RemoteWeightInfo, names: list[str]) -> None:
        source_ptrs: list[int] = []
        source_lens: list[int] = []
        valid_names: list[str] = []

        for name in names:
            cpu_reg = self._weight_memory_registry.get(name)
            if not cpu_reg:
                raise RuntimeError(
                    f"P2P source registration missing for weight {name}; "
                    "cannot transfer without corrupting rollout weights"
                )
            data_ptr, numel, ele_size = cpu_reg
            source_ptrs.append(data_ptr)
            source_lens.append(numel * ele_size)
            valid_names.append(name)

        if not source_ptrs:
            return

        session_id = remote_session.session_id
        target_ptrs = [
            remote_session.weights_info[name][0] for name in valid_names if name in remote_session.weights_info
        ]
        if len(target_ptrs) != len(source_ptrs):
            raise RuntimeError(
                f"P2P pointer count mismatch for session {session_id}: "
                f"source={len(source_ptrs)} target={len(target_ptrs)}"
            )

        ret = self._transfer_engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
        if ret < 0:
            raise RuntimeError(f"P2P transfer failed for session {session_id}, error: {ret}")

    @staticmethod
    def _import_parallelism_config(parallelism_info: Mapping[str, Any]):
        from sglang.srt.distributed.parallel_state import RankParallelismConfig

        return RankParallelismConfig.from_dict(dict(parallelism_info))

    def _create_cpu_replica(
        self,
        parallelism_config,
        model_path: str,
        server_args,
        *,
        first_engine_rank: bool = False,
    ) -> torch.nn.Module:
        from sglang.srt import server_args as server_args_module
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed.parallel_state import ParallelismContext
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
        from sglang.srt.model_loader import get_model
        from sglang.srt.model_loader import loader as model_loader_module

        load_config = LoadConfig(
            load_format="dummy",
            model_loader_extra_config=None,
            rl_quant_profile=server_args.rl_quant_profile,
        )
        server_args_module._global_server_args = server_args
        initialize_moe_config(server_args)
        initialize_fp8_gemm_config(server_args)
        initialize_fp4_gemm_config(server_args)

        # Monkey-patch loader-level post_load_weights to no-op before get_model.
        # Miles sglang uses `_post_load_weights`; sglang-pretrain exposes `post_load_weights`.
        post_load_hook_name = (
            "_post_load_weights" if hasattr(model_loader_module, "_post_load_weights") else "post_load_weights"
        )
        original_post_load_weights = getattr(model_loader_module, post_load_hook_name)
        setattr(model_loader_module, post_load_hook_name, lambda *args, **kwargs: None)
        try:
            with ParallelismContext(parallelism_config):
                model = get_model(
                    model_config=ModelConfig(model_path),
                    load_config=load_config,
                    device_config=DeviceConfig(device="cpu"),
                )
        finally:
            setattr(model_loader_module, post_load_hook_name, original_post_load_weights)

        if hasattr(model, "post_load_weights"):
            model.post_load_weights = lambda *args, **kwargs: None

        if first_engine_rank:
            for param in model.parameters():
                param.data = param.data.pin_memory()
        else:
            for name, param in model.named_parameters():
                if name not in self._shared_params_dict:
                    raise RuntimeError(f"P2P shared buffer missing parameter {name}")
                param.data = self._shared_params_dict[name]

        return model
