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
"""
Minimal checkpoint manager for DeepSpeed engines.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Optional

import torch
import torch.distributed

from verl.utils.deepspeed_utils import (
    load_deepspeed_checkpoint,
    load_deepspeed_model_to_gpu,
    offload_deepspeed_model_to_cpu,
    save_deepspeed_checkpoint,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class DeepSpeedCheckpointManager:
    """Lightweight helper around DeepSpeed checkpoint APIs."""

    CKPT_PREFIX = "step_"

    def __init__(self, engine):
        self.engine = engine
        try:
            self.rank = torch.distributed.get_rank()
        except Exception:
            self.rank = 0

    # ----- internal helpers -----
    def _ckpt_dir(self, root: str, step: int) -> str:
        return os.path.join(root, f"{self.CKPT_PREFIX}{step}")

    def _list_ckpts(self, root: str):
        if not os.path.isdir(root):
            return []
        items = []
        for name in os.listdir(root):
            if not name.startswith(self.CKPT_PREFIX):
                continue
            step_str = name[len(self.CKPT_PREFIX) :]
            if not step_str.isdigit():
                continue
            items.append((int(step_str), os.path.join(root, name)))
        return sorted(items, key=lambda kv: kv[0])

    def _latest(self, root: str):
        ckpts = self._list_ckpts(root)
        return ckpts[-1] if ckpts else None

    def _prune(self, root: str, max_keep: Optional[int]):
        if max_keep is None or max_keep <= 0:
            return
        ckpts = self._list_ckpts(root)
        if len(ckpts) <= max_keep:
            return
        stale = ckpts[:-max_keep]
        for step, path in stale:
            if self.rank == 0:
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info(f"[DeepSpeedCheckpointManager] Pruned checkpoint step={step} path={path}")
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"[DeepSpeedCheckpointManager] Failed to prune {path}: {exc}")

    # ----- public API -----
    def save_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, global_step: int = 0, max_ckpt_to_keep: Optional[int] = None
    ):
        os.makedirs(local_path, exist_ok=True)
        target = self._ckpt_dir(local_path, global_step)
        os.makedirs(target, exist_ok=True)

        load_deepspeed_model_to_gpu(self.engine)
        save_deepspeed_checkpoint(engine=self.engine, save_dir=target, tag=str(global_step))
        torch.distributed.barrier()
        offload_deepspeed_model_to_cpu(self.engine)

        if hdfs_path is not None and self.rank == 0:
            try:
                os.makedirs(hdfs_path, exist_ok=True)
                dst = self._ckpt_dir(hdfs_path, global_step)
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(target, dst)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"[DeepSpeedCheckpointManager] Failed to copy checkpoint to {hdfs_path}: {exc}")

        self._prune(local_path, max_ckpt_to_keep)

    def load_checkpoint(self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True):
        load_dir = local_path
        if os.path.isdir(local_path) and not os.path.basename(local_path).startswith(self.CKPT_PREFIX):
            latest = self._latest(local_path)
            if latest is None:
                logger.warning(f"[DeepSpeedCheckpointManager] No checkpoint found under {local_path}")
                return {}
            load_dir = latest[1]

        if hdfs_path and not os.path.exists(load_dir) and self.rank == 0:
            try:
                os.makedirs(local_path, exist_ok=True)
                shutil.copytree(hdfs_path, load_dir)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"[DeepSpeedCheckpointManager] Failed to copy checkpoint from {hdfs_path}: {exc}")

        load_deepspeed_model_to_gpu(self.engine)
        client_state = load_deepspeed_checkpoint(
            engine=self.engine,
            load_dir=load_dir,
            tag=None,
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        torch.distributed.barrier()
        offload_deepspeed_model_to_cpu(self.engine)

        if del_local_after_load and os.path.isdir(load_dir):
            try:
                shutil.rmtree(load_dir, ignore_errors=True)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"[DeepSpeedCheckpointManager] Failed to clean up {load_dir}: {exc}")
        return client_state
