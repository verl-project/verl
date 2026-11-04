# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import json
import logging
import os
import shutil
import tempfile
import time
from typing import AsyncGenerator, Generator

import torch
from safetensors import safe_open

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.utils.checkpoint.hf_safetensors import (
    HF_WEIGHTS_INDEX_NAME,
    export_hf_safetensors,
)
from verl.utils.device import get_device_name

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

INDEX_FILE = HF_WEIGHTS_INDEX_NAME
COPY_BUFFER_SIZE = 16 << 20


def _copy_file(src: str, dst: str):
    """Copy with sequential IO, which is supported by HDFS fuse mounts."""
    with open(src, "rb") as src_file, open(dst, "wb") as dst_file:
        shutil.copyfileobj(src_file, dst_file, length=COPY_BUFFER_SIZE)


@CheckpointEngineRegistry.register("filesystem")
class FileSystemCheckpointEngine(CheckpointEngine):
    """Checkpoint engine that synchronizes weights through a file system (e.g. a HDFS fuse mount), so actor
    and rollout need no direct connection and can run on different hardware or in different data centers.

    Each sync is exported as HuggingFace safetensors to ``{path}/global_step_{n}``: the shards are written
    by actor rank 0 or by all actor ranks round-robin, then rank 0 writes ``model.safetensors.index.json``.
    Rollout workers load the directory like the initial checkpoint, after which rank 0 deletes it.

    Args:
        bucket_size (int): Max size in bytes of a safetensors shard.
        path (str): Root directory of the exported weights, mounted on all actor and rollout nodes.
        distributed_write (bool): Whether all actor ranks write the shards round-robin, otherwise only
            rank 0 writes them. Defaults to True.
        is_master (bool): Whether the current process is actor rank 0.
    """

    def __init__(
        self,
        bucket_size: int,
        path: str,
        distributed_write: bool = True,
        is_master: bool = False,
    ) -> None:
        assert "://" not in path, f"path must be a mounted directory, e.g. a HDFS fuse mount, got {path}"
        self.bucket_size = bucket_size
        self.path = path
        self.distributed_write = distributed_write
        self.is_master = is_master
        self.exported_dir: str | None = None

    def prepare(self):
        return None

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list):
        return {}, {}

    def init_process_group(self):
        pass

    def finalize(self):
        """Delete the exported weights, the manager finalizes only after rollout has loaded them."""
        if self.is_master and self.exported_dir is not None:
            try:
                shutil.rmtree(self.exported_dir)
            except OSError as e:
                logger.warning(f"Failed to remove exported weights {self.exported_dir}: {e}")
            self.exported_dir = None

    def _version_dir(self, global_steps: int | None) -> str:
        return os.path.join(self.path, f"global_step_{global_steps}")

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Export the weights as safetensors shards, each writer rank writing every num_writers-th shard.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
            global_steps: The trainer step, used to name the exported directory.
        """
        rank = torch.distributed.get_rank()
        num_writers = torch.distributed.get_world_size() if self.distributed_write else 1
        version_dir = self._version_dir(global_steps)
        os.makedirs(version_dir, exist_ok=True)

        start_time = time.time()

        with tempfile.TemporaryDirectory(prefix="verl-filesystem-ckpt-") as staging_dir:
            # Every rank must drain the generator since it issues collectives; all ranks see the same
            # tensors, so they agree on the shard boundaries and the weight map.
            total_size, num_shards, _num_tensors = export_hf_safetensors(
                weights,
                output_dir=staging_dir,
                max_shard_bytes=self.bucket_size,
                rank=rank,
                num_writers=num_writers,
            )

            for file_name in sorted(os.listdir(staging_dir)):
                if not file_name.endswith(".safetensors"):
                    continue
                # local_path looks like ${TMPDIR:-/tmp}/verl-filesystem-ckpt-<random>/<file_name>
                local_path = os.path.join(staging_dir, file_name)
                # filesystem_path looks like <path>/global_step_<n>/<file_name>
                filesystem_path = os.path.join(version_dir, file_name)
                _copy_file(local_path, filesystem_path)

            # Copy the index last to commit the version only after every shard has landed.
            await asyncio.to_thread(torch.distributed.barrier)
            if self.is_master:
                _copy_file(os.path.join(staging_dir, INDEX_FILE), os.path.join(version_dir, INDEX_FILE))
                self.exported_dir = version_dir

        time_cost = time.time() - start_time
        logger.info(
            f"Rank {rank} export weights to {version_dir} done, total size: {total_size / (1 << 30):.2f} GB, "
            f"shards: {num_shards}, time cost: {time_cost:.2f}s"
        )

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Load the weights exported for ``global_steps``.

        Args:
            global_steps: The trainer step, used to locate the exported directory.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        start_time = time.time()
        version_dir = self._version_dir(global_steps)
        with open(os.path.join(version_dir, INDEX_FILE)) as f:
            weight_map = json.load(f)["weight_map"]

        device = get_device_name()
        for file_name in sorted(set(weight_map.values())):
            with safe_open(os.path.join(version_dir, file_name), framework="pt") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name).to(device)

        logger.info(f"Load weights from {version_dir} done, time cost: {time.time() - start_time:.2f}s")
