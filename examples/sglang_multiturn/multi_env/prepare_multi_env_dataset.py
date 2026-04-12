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

#!/usr/bin/env python3
"""Generate train/val parquets for the multi-env tool-use example.

Each row carries ``extra_info.tool_env_id`` and ``extra_info.db_path`` so that
the ``MultiEnvToolAgentLoop`` can resolve per-instance tools and shared DB.

Usage::

    python prepare_multi_env_dataset.py --output_dir ./data
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCENARIOS = [
    {
        "data_source": "multi_env_demo/inventory",
        "tool_env_id": "inventory_store",
        "db_file": "inventory_001.json",
        "system": "You are a shopping assistant with access to a check_stock tool.",
        "user": "Is the laptop in stock? What's the price?",
    },
    {
        "data_source": "multi_env_demo/inventory",
        "tool_env_id": "inventory_store",
        "db_file": "inventory_001.json",
        "system": "You are a shopping assistant with access to a check_stock tool.",
        "user": "How many phones do you have available?",
    },
    {
        "data_source": "multi_env_demo/bank",
        "tool_env_id": "bank_account",
        "db_file": "bank_001.json",
        "system": "You are a banking assistant with access to a check_balance tool.",
        "user": "What is Alice's current balance?",
    },
    {
        "data_source": "multi_env_demo/bank",
        "tool_env_id": "bank_account",
        "db_file": "bank_001.json",
        "system": "You are a banking assistant with access to a check_balance tool.",
        "user": "Check Bob's account balance for me.",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--db_dir", default="./sample_dbs")
    args = parser.parse_args()

    db_dir = str(Path(args.db_dir).resolve())
    rows: list[dict] = []
    for idx, s in enumerate(SCENARIOS):
        rows.append(
            {
                "data_source": s["data_source"],
                "prompt": json.dumps(
                    [{"role": "system", "content": s["system"]}, {"role": "user", "content": s["user"]}],
                    ensure_ascii=False,
                ),
                "ability": "tool_use",
                "reward_model": json.dumps({"style": "rule", "ground_truth": ""}),
                "extra_info": json.dumps(
                    {"index": idx, "tool_env_id": s["tool_env_id"], "db_path": os.path.join(db_dir, s["db_file"])},
                    ensure_ascii=False,
                ),
            }
        )

    train_rows, val_rows = rows[:-1], rows[-1:]

    os.makedirs(args.output_dir, exist_ok=True)
    for split, split_rows in [("train", train_rows), ("val", val_rows)]:
        table = pa.table({k: [r[k] for r in split_rows] for k in split_rows[0]})
        path = os.path.join(args.output_dir, f"{split}.parquet")
        pq.write_table(table, path)
        print(f"Wrote {len(split_rows)} rows -> {path}")


if __name__ == "__main__":
    main()
