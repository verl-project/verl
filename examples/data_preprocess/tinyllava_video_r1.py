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
Preprocess Zhang199/TinyLLaVA-Video-R1-training-data to verl parquet format.

  - Prompt: "<video>{problem}" (problem already has inline options)
  - Video: absolute file path
  - Label: solution ("<answer>X</answer>")
  - Uses verl-standard inline instruction (think/answer format).

Usage:
    # Step 1: Download
    export HF_ENDPOINT=https://hf-mirror.com
    hf download Zhang199/TinyLLaVA-Video-R1-training-data \\
        --repo-type dataset --local-dir ~/data/tinyllava-video-r1

    # Step 2: Extract videos
    unzip ~/data/tinyllava-video-r1/NextQA.zip -d ~/data/tinyllava-video-r1/

    # Step 3: Preprocess
    python examples/data_preprocess/tinyllava_video_r1.py \\
        --data_dir ~/data/tinyllava-video-r1 \\
        --local_save_dir ~/data/tinyllava_video_r1
"""

import argparse
import json
import os
import sys
from typing import Optional

import datasets

from verl.utils.hdfs_io import copy, makedirs

DATA_SOURCE = "Zhang199/TinyLLaVA-Video-R1-training-data"

# Inline instruction: think/answer format for video QA.
INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue "
    "and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST be a single option letter (e.g., A, B, C, D, E) "
    "enclosed within <answer> </answer> tags."
)


def build_prompt_text(problem: str) -> str:
    """Build prompt with video placeholder: "<video>{problem}".

    The JSONL problem field already contains inline options:
      "What animal is shown?\nOptions:\nA. owl.\nB. sheeps.\n..."
    """
    return f"<video>\n{problem}\n\n{INSTRUCTION}"


def make_map_fn(
    data_source: str,
    video_dir: str,
    split: str,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
):
    """Factory function following verl geo3k/openr1mm closure pattern."""

    def process_fn(example, idx):
        problem = example["problem"]
        solution = example["solution"]  # already "<answer>X</answer>"

        # Resolve video path from video_dir + video_filename.
        # JSONL paths are "./NextQA/NExTVideo/..." → strip "./" prefix.
        video_rel = example["video_filename"].lstrip("./")
        video_path = os.path.join(video_dir, video_rel)
        if not os.path.exists(video_path):
            print(f"[WARN] Video file not found: {video_path}", file=sys.stderr)

        prompt_content = build_prompt_text(problem)

        # Video sampling params (fps=1, max_frames=32)
        video_entry = {"video": video_path}
        if video_fps is not None:
            video_entry["fps"] = video_fps
        if video_max_frames is not None:
            video_entry["max_frames"] = video_max_frames

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_content}],
            "videos": [video_entry],
            "ability": "video_qa",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": problem,
                "answer": solution,
                "video_path": video_path,
            },
        }

    return process_fn


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess TinyLLaVA-Video-R1 to verl parquet.")
    parser.add_argument("--data_dir", type=str, default=None, help="Downloaded dataset directory.")
    parser.add_argument("--local_save_dir", default="~/data/tinyllava_video_r1", help="Output directory.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--video_fps", type=float, default=1, help="Video sampling FPS (default: 1)")
    parser.add_argument("--video_max_frames", type=int, default=32, help="Max frames per video (default: 32)")
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("--data_dir is required")

    # ---- Load ----
    jsonl_path = os.path.join(args.data_dir, "nextqa_0-30s.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"[ERROR] Not found: {jsonl_path}")
        sys.exit(1)

    print(f"Loading: {jsonl_path}")
    data = load_jsonl(jsonl_path)
    print(f"  {len(data)} samples")

    # Sanity check
    s0 = data[0]
    print(f"  First sample problem: {s0['problem'][:80]}...")
    print(f"  First sample video:   {s0['video_filename']}")
    print(f"  First sample solution: {s0['solution']}")

    # ---- Video directory ----
    # JSONL video_filename includes "NextQA/" prefix (e.g. "./NextQA/NExTVideo/..."),
    # so video_dir must be the dataset root, not dataset_root/NextQA.
    video_dir = args.data_dir

    # ---- Convert + 90/10 split (same as openr1mm.py) ----
    full_dataset = datasets.Dataset.from_list(data)
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)

    train_map_fn = make_map_fn(
        DATA_SOURCE, video_dir, "train", video_fps=args.video_fps, video_max_frames=args.video_max_frames
    )
    test_map_fn = make_map_fn(
        DATA_SOURCE, video_dir, "test", video_fps=args.video_fps, video_max_frames=args.video_max_frames
    )
    train_dataset = split_dataset["train"].map(function=train_map_fn, with_indices=True, num_proc=4)
    test_dataset = split_dataset["test"].map(function=test_map_fn, with_indices=True, num_proc=4)

    columns = ["data_source", "prompt", "videos", "ability", "reward_model", "extra_info"]
    train_dataset = train_dataset.select_columns(columns)
    test_dataset = test_dataset.select_columns(columns)

    # ---- Save ----
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_out = os.path.join(save_dir, "train.parquet")
    test_out = os.path.join(save_dir, "test.parquet")

    print(f"\nSaving train ({len(train_dataset)} samples) → {train_out}")
    train_dataset.to_parquet(train_out)
    print(f"Saving test  ({len(test_dataset)} samples) → {test_out}")
    test_dataset.to_parquet(test_out)

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=save_dir, dst=args.hdfs_dir)

    print(f"\nDone! Train: {train_out}, Test: {test_out}")
