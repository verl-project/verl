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
"""
Create a small synthetic parquet dataset for FlowGRPO diffusion e2e testing.

The dataset uses the jpeg_compressibility reward (a self-contained rule-based
reward that needs no external reward model) so the e2e test can run without
spinning up a separate vLLM reward server.
"""

import argparse
import os

import pandas as pd

SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)

USER_PROMPTS = [
    "A red circle on a white background",
    "A blue square on a black background",
    "A green triangle next to an orange rectangle",
    "The word HELLO written in bold letters",
    "A yellow star above a purple crescent moon",
    "Two overlapping circles, one red and one blue",
    "A gradient from dark blue to light blue",
    "A checkerboard pattern of black and white squares",
]


def build_rows(split: str, n: int):
    rows = []
    for i in range(n):
        prompt_text = USER_PROMPTS[i % len(USER_PROMPTS)]
        rows.append(
            {
                "data_source": "jpeg_compressibility",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "negative_prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": " "},
                ],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": split, "index": i},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate dummy diffusion parquet data for e2e testing")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/dummy_diffusion"),
        help="Directory to write train.parquet and test.parquet",
    )
    parser.add_argument("--train_size", type=int, default=32, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=8, help="Number of validation samples")
    args = parser.parse_args()

    os.makedirs(args.local_save_dir, exist_ok=True)

    train_df = pd.DataFrame(build_rows("train", args.train_size))
    val_df = pd.DataFrame(build_rows("test", args.val_size))

    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "test.parquet")

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")


if __name__ == "__main__":
    main()
