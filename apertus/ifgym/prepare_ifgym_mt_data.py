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
"""Build train/test parquet files for IFGym multi-turn RL.

Reads the swiss-ai/if-gym ``v2-multiturn-if`` dataset layout
(``<src>/<split>/data.jsonl``, each line a multi-turn sample) and emits verl
parquet files. Each row carries:
  * ``prompt``: the first user turn (chat format)
  * ``extra_info.interaction_kwargs.turns_json``: JSON list of all turns, each
    ``{"prompt", "active_constraints": [{"constraint_id", "kwargs"}, ...]}``.
The IFGym agent loop (``ifgym_agent``) scripts the remaining user turns and
scores each assistant turn against that turn's ``active_constraints``.
"""

import argparse
import json
import os

import pandas as pd


def convert_split(src: str, split: str, n_max: float) -> pd.DataFrame:
    rows = []
    with open(f"{src}/{split}/data.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n_max:
                break
            s = json.loads(line)
            all_turns = [
                {
                    "prompt": t.get("rendered_prompt", t.get("task_text", "")),
                    "active_constraints": [
                        {"constraint_id": c["constraint_id"], "kwargs": c.get("kwargs") or {}}
                        for c in t.get("active_constraints", [])
                    ],
                }
                for t in s["turns"]
            ]
            if not all_turns:
                continue
            rows.append(
                {
                    "data_source": "ifgym_multiturn",
                    "prompt": [{"role": "user", "content": all_turns[0]["prompt"]}],
                    "ability": "instruction_following",
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": {
                        "interaction_kwargs": {
                            "name": "ifgym",
                            "turns_json": json.dumps(all_turns),
                        }
                    },
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IFGym multi-turn train/test parquet files.")
    parser.add_argument(
        "--src",
        required=True,
        help="Source dir containing <split>/data.jsonl (e.g. the v2-multiturn-if dataset).",
    )
    parser.add_argument("--out", required=True, help="Output dir for train.parquet / test.parquet.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--n-train", type=float, default=float("inf"))
    parser.add_argument("--n-val", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train_df = convert_split(args.src, args.train_split, args.n_train)
    val_df = convert_split(args.src, args.val_split, args.n_val)
    train_df.to_parquet(f"{args.out}/train.parquet")
    val_df.to_parquet(f"{args.out}/test.parquet")
    print(f"Wrote {len(train_df)} train + {len(val_df)} val to {args.out}")


if __name__ == "__main__":
    main()
