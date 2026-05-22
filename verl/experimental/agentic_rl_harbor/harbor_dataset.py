# Copyright 2026 Alibaba Group
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
Dataset that yields Harbor task directory paths.

Each item is one Harbor task (a directory containing ``instruction.md``).
The actual task instruction is loaded lazily by Harbor inside the agent loop;
we only carry the path here.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HarborTaskDataset(Dataset):
    """List Harbor task directories under ``data_files``.

    A task directory is any subdirectory containing an ``instruction.md`` file.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer=None,
        processor=None,
        config: Optional[DictConfig] = None,
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        if isinstance(data_files, str):
            data_files = [data_files]

        task_paths: list[Path] = []
        for source in data_files:
            root = Path(source)
            if not root.exists():
                logger.warning("Harbor data path does not exist: %s", source)
                continue
            if root.is_dir():
                if self._is_task_dir(root):
                    task_paths.append(root)
                    continue
                task_paths.extend(sorted(d for d in root.iterdir() if self._is_task_dir(d)))
            else:
                logger.warning("Harbor data path is not a directory: %s", source)

        if max_samples > 0:
            task_paths = task_paths[:max_samples]

        if not task_paths:
            raise ValueError(f"No Harbor tasks found under {data_files}")

        self.task_paths = task_paths
        logger.info("HarborTaskDataset initialized with %d tasks", len(self.task_paths))

    @staticmethod
    def _is_task_dir(path: Path) -> bool:
        return path.is_dir() and (path / "instruction.md").is_file()

    def __len__(self) -> int:
        return len(self.task_paths)

    def __getitem__(self, idx: int) -> dict:
        task_path = str(self.task_paths[idx])
        return {
            # raw_prompt is unused by HarborAgentLoop (Harbor reads the task itself),
            # but VeRL's pipeline expects it to be present.
            "raw_prompt": [{"role": "user", "content": task_path}],
            "task_path": task_path,
            "agent_name": "harbor_agent",
            "data_source": "harbor",
            "reward_model": {"ground_truth": ""},
            "extra_info": {"index": idx, "task_path": task_path},
            "index": idx,
            "tools_kwargs": {},
            "interaction_kwargs": {},
            # DataProto.batch must be non-empty.
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }
