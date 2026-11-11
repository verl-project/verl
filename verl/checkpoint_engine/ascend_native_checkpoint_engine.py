# Copyright 2026 Individual Contributor
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
import hashlib
import logging
from functools import cached_property

import ray
import torch

from verl.checkpoint_engine.base import CheckpointEngineRegistry, DirectRolloutCheckpointEngine
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)


def _make_source(model_engine):
    # Only the trainer process needs the vLLM trainer API.
    from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource

    class FSDPWeightSource(WeightSource):
        @cached_property
        def metadata(self):
            return [ParamMeta(name, dtype, shape) for name, dtype, shape in model_engine.get_weight_transfer_metadata()]

        def iter_full(self):
            # Ascend's packed producer advances this iterator on its own rotating NPU streams.
            # Keep FSDP gathers and any reused exporter storage ordered on the trainer stream.
            device = get_torch_device()
            with device.stream(self.export_stream):
                weights, _ = model_engine.get_per_tensor_param()
                weights = iter(weights)
            while True:
                with device.stream(self.export_stream):
                    try:
                        name, weight = next(weights)
                    except StopIteration:
                        return
                # FSDP2 full tensors (including MoE views) retain their own storage.
                # Keep it live until the native consumer has read it on its stream.
                consumer = device.current_stream()
                if weight.device.type == "npu" and consumer != self.export_stream:
                    consumer.wait_stream(self.export_stream)
                    weight.record_stream(consumer)
                yield name, weight

    return FSDPWeightSource()


@CheckpointEngineRegistry.register("vllm_native_hccl")
class AscendNativeCheckpointEngine(DirectRolloutCheckpointEngine):
    """Trainer endpoint for dense weight updates to disaggregated vLLM-Ascend workers."""

    def __init__(self, bucket_size: int, is_master: bool = False, packed: bool = True):
        if bucket_size <= 0:
            raise ValueError("vllm_native_hccl requires a positive update bucket size")
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.packed = packed
        self.generation = None
        self._connected_generation = None
        self._trainer = None
        self._source = None

    @staticmethod
    def get_rollout_endpoints(replicas):
        from verl.workers.rollout.vllm_rollout.native_weight_sync import get_native_rollout_endpoints

        return get_native_rollout_endpoints(replicas)

    @staticmethod
    async def finish_rollout_update(replicas, global_steps):
        from verl.workers.rollout.vllm_rollout.native_weight_sync import finish_native_rollout_update

        return await finish_native_rollout_update(replicas, global_steps)

    def prepare(self):
        if get_device_name() != "npu":
            raise ValueError("vllm_native_hccl requires Ascend NPU trainer workers")
        if self.is_master:
            address = ray.util.get_node_ip_address().strip("[]")
            port, _ = get_free_port(address)
            return address, port
        return None

    def configure(self, master_address, master_port, world_size, endpoints, generation):
        self.master_address = master_address
        self.master_port = master_port
        self.world_size = world_size
        self.endpoints = endpoints
        self.generation = generation
        # A replacement manager can restart its generation counter while an
        # existing trainer still holds the previous manager's communicator.
        self._connected_generation = None

    def _open(self, model_engine):
        # Only enter the trainer-to-rollout rendezvous after all FSDP ranks have
        # confirmed they can replay the same export. A rank-local exception later
        # would strand its peers in their first collective.
        try:
            from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
            from vllm.plugins import load_general_plugins
            from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

            from verl.workers.rollout.vllm_rollout.native_weight_sync import RayWeightSyncClient

            # The Ascend general plugin registers its native HCCL trainer and worker.
            load_general_plugins()
            source = _make_source(model_engine)
            fields = [(m.name, str(m.dtype), m.shape) for m in source.metadata]
            if not fields:
                raise ValueError("Native HCCL requires a nonempty full-weight export")
            signature = hashlib.sha256(repr(fields).encode()).hexdigest()
            client = RayWeightSyncClient(self.endpoints)
            info = HCCLTrainerInitInfo(
                rank=torch.distributed.get_rank(),
                master_address=self.master_address,
                master_port=self.master_port,
                world_size=self.world_size,
                packed=self.packed,
                packed_buffer_size_bytes=self.bucket_size,
            )
            status = (None, signature)
        except Exception as exc:
            status = (f"{type(exc).__name__}: {exc}", None)

        statuses = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(statuses, status)
        for rank, (error, _) in enumerate(statuses):
            if error:
                raise RuntimeError(f"Native HCCL trainer preflight failed on rank {rank}: {error}")
        if any(schema != statuses[0][1] for _, schema in statuses):
            raise ValueError("Native HCCL requires identical full-weight metadata on every trainer rank")

        self._trainer = WeightTransferTrainerFactory.trainer_init(info, client=client, source=source)
        self._source = source
        self._connected_generation = self.generation

    def _send(self, model_engine, device_index, stream):
        device = get_torch_device()
        with torch.no_grad(), device.device(device_index), device.stream(stream):
            if self._connected_generation != self.generation:
                if self._trainer is not None:
                    self._trainer.shutdown()
                    self._trainer = None
                self._open(model_engine)
            self._source.export_stream = stream
            try:
                self._trainer.send_weights()
            except BaseException:
                # A partially completed native transaction cannot be retried on the same group.
                trainer, self._trainer = self._trainer, None
                self._connected_generation = None
                self._source = None
                try:
                    trainer.shutdown()
                except Exception:
                    logger.exception("Could not close failed native HCCL trainer")
                raise

    async def send_weights(self, model_engine, global_steps=None):
        if self.generation is None:
            raise RuntimeError("Configure the native rollout endpoints before sending weights")
        device = get_torch_device()
        # The synchronous native engine waits on worker RPCs and enters HCCL rendezvous.
        # Capture device/stream on the Ray actor before crossing the thread boundary.
        transfer = asyncio.create_task(
            asyncio.to_thread(self._send, model_engine, device.current_device(), device.current_stream())
        )
        try:
            await asyncio.shield(transfer)
        except asyncio.CancelledError:
            # The thread still owns trainer parameters, possibly in a collective. Wait
            # before allowing the caller to offload or resume training on this model.
            await transfer
            raise
