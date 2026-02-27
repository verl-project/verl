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

import argparse
import glob
import logging
import os
import sys

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def check_profiler_output(profiler_dir: str) -> bool:
    """Check if profiler deliverables are generated correctly"""
    logger.info("Starting profiler deliverables check...")

    # Check if profiler_data directory exists
    if not os.path.exists(profiler_dir):
        logger.error(f"Profiler data directory not found: {profiler_dir}")
        return False

    target_stages = ["actor_update", "*_rollout_*", "ref_*"]

    for stage in target_stages:
        if stage == "*_rollout_*":
            expected_count = 2
        else:
            expected_count = 1

        # Find all xxx_ascend_xxx directories
        search_pattern = os.path.join(profiler_dir, stage, "*_ascend_*")
        dirs = glob.glob(search_pattern, recursive=True)

        # Print found directories for debugging
        for d in dirs:
            print(f"[{stage}] Found: {d}")

        if len(dirs) != expected_count:
            logger.error(f"[{stage}] Expected {expected_count} *_ascend_* directories, found {len(dirs)}")
            return False

        # Check each xxx_ascend_xxx directory
        for ascend_dir in dirs:
            profiler_output = glob.glob(os.path.join(ascend_dir, "PROF_*"))
            if not profiler_output or not os.path.isdir(profiler_output[0]):
                logger.error(f"[{stage}] PROF not found in {ascend_dir}")
                return False

    logger.info("All profiler deliverables check passed!")
    return True


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check and clean Profiler deliverables")
    parser.add_argument("--profiler-dir", type=str, default="./profiler_data", help="Path to profiler data directory")
    return parser.parse_args()


def main():
    args = parse_args()

    if check_profiler_output(args.profiler_dir):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
