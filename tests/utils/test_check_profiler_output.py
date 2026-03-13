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
from dataclasses import dataclass
from typing import Callable

# Initialize logger
logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@dataclass
class DeviceCheckConfig:
    """Device check configuration: encapsulates device-specific validation rules"""

    # Search path pattern
    search_pattern: str
    # Directory count validation function: takes stage and dir list, returns bool
    dir_count_validator: Callable[[str, list[str]], bool]
    # PROF file/dir validation function: takes directory path, returns bool
    prof_validator: Callable[[str], bool]


class ProfilerChecker:
    """Unified Profiler checker supporting GPU/NPU devices"""

    TARGET_STAGES = ["actor_update", "*_rollout_*", "ref_*"]

    def __init__(self, device_type: str, profiler_dir: str):
        self.device_type = device_type.lower()
        self.profiler_dir = profiler_dir

        # Validate device type
        if self.device_type not in ["gpu", "npu"]:
            raise ValueError(f"Unsupported device type: {device_type}, only gpu/npu are supported")

        # Initialize device-specific configuration
        self._init_device_config()

    def _init_device_config(self):
        """Initialize validation rules for different devices (core: device differences as config)"""
        if self.device_type == "gpu":
            self.config = DeviceCheckConfig(
                # GPU search pattern: match stage directory directly
                search_pattern=os.path.join(self.profiler_dir, "{stage}"),
                # GPU: all stages must have exactly 1 directory
                dir_count_validator=lambda stage, dirs: len(dirs) == 1,
                # GPU: any file/subdirectory exists under the directory
                prof_validator=lambda d: len(glob.glob(os.path.join(d, "*"))) > 0,
            )
        else:  # NPU
            self.config = DeviceCheckConfig(
                # NPU search pattern: match ascend subdirectory under stage
                search_pattern=os.path.join(self.profiler_dir, "{stage}", "*_ascend_*"),
                # NPU: rollout requires >1 dir, others require exactly 1 dir
                dir_count_validator=lambda stage, dirs: (len(dirs) > 1 if stage == "*_rollout_*" else len(dirs) == 1),
                # NPU: PROF_* subdirectory must exist and be a valid directory
                prof_validator=lambda d: (
                    len(glob.glob(os.path.join(d, "PROF_*"))) > 0
                    and os.path.isdir(glob.glob(os.path.join(d, "PROF_*"))[0])
                ),
            )

    def _validate_stage_dirs(self, stage: str) -> bool:
        """Generic stage directory validation: extracted common logic for GPU/NPU"""
        # 1. Generate search path and match directories
        search_pattern = self.config.search_pattern.format(stage=stage)
        dirs = glob.glob(search_pattern, recursive=True)

        # 2. Log found directories
        for d in dirs:
            logger.info(f"[{stage}] Found: {d}")

        # 3. Validate directory count
        if not self.config.dir_count_validator(stage, dirs):
            expected = ">1" if stage == "*_rollout_*" and self.device_type == "npu" else 1
            logger.error(f"[{stage}] Expected {expected} directories, found {len(dirs)}")
            return False

        # 4. Validate PROF files/directories
        for target_dir in dirs:
            if not self.config.prof_validator(target_dir):
                logger.error(f"[{stage}] PROF not found in {target_dir}")
                return False

        return True

    def check(self) -> bool:
        """Unified check entry point"""
        logger.info(f"Starting profiler deliverables check for {self.device_type.upper()}...")

        # Validate root directory exists
        if not os.path.exists(self.profiler_dir):
            logger.error(f"Profiler data directory not found: {self.profiler_dir}")
            return False

        # Run validation for all target stages
        for stage in self.TARGET_STAGES:
            if not self._validate_stage_dirs(stage):
                return False

        logger.info(f"All {self.device_type.upper()} validation stages passed")
        return True


def parse_args():
    """Parse command line arguments"""
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
