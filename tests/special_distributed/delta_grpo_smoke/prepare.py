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
"""Prepare fixed synthetic inputs for a delta GRPO smoke run."""

import os
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

model = os.environ["MODEL_PATH"]
work = Path(os.environ["SMOKE_DIR"])
work.mkdir(parents=True, exist_ok=True)
AutoTokenizer.from_pretrained(model, local_files_only=True)
rows = [
    dict(
        data_source="delta_smoke",
        prompt=[dict(role="user", content=f"Calculate {i % 8}+{(i + 1) % 8}. Reply with the result.")],
        ability="math",
        reward_model=dict(style="rule", ground_truth=str(i % 8 + (i + 1) % 8)),
        extra_info=dict(index=i),
    )
    for i in range(32)
]
pd.DataFrame(rows).to_parquet(work / "train.parquet")
pd.DataFrame(rows[:4]).to_parquet(work / "val.parquet")
print("INPUTS_OK", model)
