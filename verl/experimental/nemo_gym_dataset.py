# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class NemoGymJSONLDataset(Dataset):
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer,
        processor=None,
        config=None,
        **kwargs,
    ):
        if isinstance(data_files, str):
            data_files = [data_files]

        self.tokenizer = tokenizer
        self._rows: list[dict] = []

        for path in data_files:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"NemoGymJSONLDataset: file not found: {path}")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]

        rcp = row.get("responses_create_params", {})
        messages = rcp.get("input", [])
        raw_prompt = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]

        agent_ref = row.get("agent_ref", None)

        skip_keys = {"responses_create_params", "agent_ref"}
        extra_env_info = {k: v for k, v in row.items() if k not in skip_keys}

        # preserve per-task fields like `tools` to nemo-gym request
        # input/temperature/top_p/top_k are overridden per training step
        # parallel_tool_calls is dropped because vLLM throws 500 error
        _rcp_skip = {"input", "temperature", "top_p", "top_k", "parallel_tool_calls"}
        rcp_extra = {k: v for k, v in rcp.items() if k not in _rcp_skip}
        if rcp_extra:
            extra_env_info["_rcp_extra"] = rcp_extra

        out = {
            "raw_prompt": raw_prompt,
            # unused placeholder. ray_trainer.py calls len(batch.batch) which requires batch to be non-empty
            "__nemo_gym_batch_size__": torch.zeros(1, dtype=torch.long),
        }
        if agent_ref is not None:
            out["agent_ref"] = agent_ref
        out["extra_env_info"] = extra_env_info

        return out

    @property
    def collate_fn(self):
        from verl.utils.dataset.rl_dataset import collate_fn

        return collate_fn
