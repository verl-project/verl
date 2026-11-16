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
"""Build the caesar_cipher + simple_geometry prompts for multi-teacher OPD on reasoning_gym.

Generates the same prompts as Miles' `examples/on_policy_distillation/mopd_reasoning_gym/prepare_data.py`
and the teacher cards (semianalysisai/Qwen3.6-35B-A3B-{caesar-cipher,simple-geometry}-GRPO): reasoning_gym
0.1.25 default task configs, 10,000 train prompts per task from seed 20000 (mixed and shuffled with
random.Random(0)), 128 held-out prompts per task from seed 10000, reasoning_gym's default system prompt.
Prompts longer than --max-prompt-tokens are dropped when --tokenizer is given (none are at 2,048).

Writes, under --out:
    train.parquet                     both tasks; `data_source` is the task and routes to its teacher
    caesar_cipher-dev.parquet         held-out
    simple_geometry-dev.parquet       held-out
    manifest.json                     sizes, seeds, and sha256 of every file

Example:
    pip install reasoning_gym==0.1.25
    python examples/on_policy_distillation_trainer/qwen3_6_35b_mopd/prepare_data.py \
        --out ~/data/mopd_reasoning_gym --tokenizer Qwen/Qwen3.6-35B-A3B
"""

import argparse
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TASKS = ("caesar_cipher", "simple_geometry")
TRAIN_SEED = 20000
EVAL_SEED = 10000


def _prompt_tokens(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    return len(encoded["input_ids"] if isinstance(encoded, Mapping) else encoded)


def _rows(task, size, seed, tokenizer, max_prompt_tokens):
    import reasoning_gym
    from reasoning_gym.utils import SYSTEM_PROMPTS

    rows, dropped = [], 0
    for index, entry in enumerate(reasoning_gym.create_dataset(task, size=size, seed=seed)):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["default"]},
            {"role": "user", "content": entry["question"]},
        ]
        if tokenizer is not None and _prompt_tokens(tokenizer, messages) > max_prompt_tokens:
            dropped += 1
            continue
        # The scorer (reward.py) rebuilds the reasoning_gym entry from the ground truth.
        label = {"task": task, "seed": seed, "index": index, **entry}
        rows.append(
            {
                "prompt": messages,
                "data_source": task,
                "ability": task,
                "reward_model": {"style": "rule", "ground_truth": json.dumps(label)},
                "extra_info": {"source_seed": seed, "source_index": index},
            }
        )
    return rows, dropped


def _write(path, rows):
    for index, row in enumerate(rows):
        row["extra_info"] = {"index": index, **row["extra_info"]}
    pq.write_table(pa.Table.from_pylist(rows), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default=None, help="HF checkpoint used to drop prompts over the limit")
    parser.add_argument("--train-size", type=int, default=10000, help="Train prompts generated per task")
    parser.add_argument("--eval-size", type=int, default=128, help="Held-out prompts per task")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"reasoning_gym": "0.1.25", "train_seed": TRAIN_SEED, "eval_seed": EVAL_SEED, "files": {}}

    train = []
    for task in TASKS:
        rows, dropped = _rows(task, args.train_size, TRAIN_SEED, tokenizer, args.max_prompt_tokens)
        train += rows
        manifest[f"train_{task}"] = {"rows": len(rows), "dropped_over_limit": dropped}
        rows, dropped = _rows(task, args.eval_size, EVAL_SEED, tokenizer, args.max_prompt_tokens)
        name = f"{task}-dev.parquet"
        manifest["files"][name] = _write(out / name, rows)
        manifest[f"eval_{task}"] = {"rows": len(rows), "dropped_over_limit": dropped}

    random.Random(0).shuffle(train)
    manifest["files"]["train.parquet"] = _write(out / "train.parquet", train)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
