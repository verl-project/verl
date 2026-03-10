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

# Initialize logger
logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class ProfilerChecker:
    """Unified Profiler checker supporting GPU/NPU devices"""

    TARGET_STAGES = ["actor_update", "*_rollout_*", "ref_*"]

    def __init__(self, device_type: str, profiler_dir: str):
        self.device_type = device_type.lower()  # Convert to lowercase uniformly to avoid case issues
        self.profiler_dir = profiler_dir

        # Validate device type legality
        if self.device_type not in ["gpu", "npu"]:
            raise ValueError(f"Unsupported device type: {device_type}, only gpu/npu are supported")

    def _check_gpu_profiler(self) -> bool:
        """GPU version of Profiler check logic"""
        for stage in self.TARGET_STAGES:
            # Build search path
            search_pattern = os.path.join(self.profiler_dir, stage)
            dirs = glob.glob(search_pattern, recursive=True)

            # Print debug information
            for d in dirs:
                print(f"[{stage}] Found: {d}")

            # Check directory count
            if len(dirs) != 1:
                logger.error(f"[{stage}] Expected 1 directories, found {len(dirs)}")
                return False

            # Check files in directory
            for gpu_dir in dirs:
                if "_rollout_" in gpu_dir:
                    expected_count = 3
                else:
                    expected_count = 1

                profiler_output = glob.glob(os.path.join(gpu_dir, "*"))
                if not profiler_output or len(profiler_output) != expected_count:
                    logger.error(f"[{stage}] PROF not found in {gpu_dir}")
                    return False

        return True

    def _check_npu_profiler(self) -> bool:
        """NPU version of Profiler check logic"""
        for stage in self.TARGET_STAGES:
            # Determine expected directory count for each stage
            if stage == "*_rollout_*":
                expected_count = 2
            else:
                expected_count = 1

            # Build NPU-specific path (xxx_ascend_xxx)
            search_pattern = os.path.join(self.profiler_dir, stage, "*_ascend_*")
            dirs = glob.glob(search_pattern, recursive=True)

            # Print debug information
            for d in dirs:
                print(f"[{stage}] Found: {d}")

            # Check directory count
            if len(dirs) != expected_count:
                logger.error(f"[{stage}] Expected {expected_count} *_ascend_* directories, found {len(dirs)}")
                return False

            # Check PROF files in NPU directory
            for ascend_dir in dirs:
                profiler_output = glob.glob(os.path.join(ascend_dir, "PROF_*"))
                if not profiler_output or not os.path.isdir(profiler_output[0]):
                    logger.error(f"[{stage}] PROF not found in {ascend_dir}")
                    return False
        return True

    def check(self) -> bool:
        """Unified check entry"""
        logger.info(f"Starting profiler deliverables check for {self.device_type.upper()}...")

        # First check if root directory exists
        if not os.path.exists(self.profiler_dir):
            logger.error(f"Profiler data directory not found: {self.profiler_dir}")
            return False

        # Call corresponding check logic based on device type
        if self.device_type == "gpu":
            return self._check_gpu_profiler()
        else:  # npu
            return self._check_npu_profiler()


def parse_args():
    """Parse command line arguments (add device parameter)"""
    parser = argparse.ArgumentParser(description="Check Profiler deliverables (support GPU/NPU)")
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["gpu", "npu"],
        help="Device type, available values: gpu/npu (required)",
    )
    parser.add_argument(
        "--profiler_dir",
        type=str,
        default="./profiler_data",
        help="Path to profiler data directory (default: ./profiler_data)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        # Initialize checker and execute check
        checker = ProfilerChecker(device_type=args.device, profiler_dir=args.profiler_dir)
        if checker.check():
            logger.info(f"All {args.device.upper()} profiler deliverables check passed!")
            sys.exit(0)
        else:
            logger.error(f"{args.device.upper()} profiler check failed!")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Check failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
