#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import Dataset, load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = SCRIPT_DIR / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_gsm8k.verl_dynamic_mc import make_stage1_records_from_gsm8k_example


def main() -> None:
    parser = argparse.ArgumentParser(description="Create VERL-native dynamic MC Stage 1 seed parquet.")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--num-samples", type=int, default=None, help="Optional GSM8K question cap.")
    parser.add_argument(
        "--stage1-prompt-count",
        type=int,
        default=None,
        help="Neutral candidate-generation prompts per question. Defaults to 4.",
    )
    parser.add_argument(
        "--incorrect-target-count",
        type=int,
        default=None,
        help="Deprecated alias: neutral mode uses this as stage1_prompt_count - 1.",
    )
    parser.add_argument("--prompt-mode", choices=["neutral", "role"], default="neutral")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    args = parser.parse_args()

    if args.stage1_prompt_count is not None:
        seed_count = max(1, args.stage1_prompt_count)
    elif args.incorrect_target_count is not None:
        seed_count = max(1, args.incorrect_target_count + 1)
    else:
        seed_count = 4
    role_incorrect_count = max(0, seed_count - 1)

    source = load_dataset("openai/gsm8k", "main")[args.split]
    if args.num_samples is not None:
        source = source.select(range(min(args.num_samples, len(source))))

    records = []
    for index, example in enumerate(source):
        records.extend(
            make_stage1_records_from_gsm8k_example(
                split=args.split,
                index=index,
                example=example,
                incorrect_target_count=role_incorrect_count,
                prompt_mode=args.prompt_mode,
            )
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(output_path))
    print(
        f"[create_dynamic_mc_seed] split={args.split} questions={len(source)} records={len(records)} "
        f"prompt_mode={args.prompt_mode} stage1_prompt_count={seed_count} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
