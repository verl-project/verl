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

    search_patterns: Callable[[str], list[str]]
    path_filter: Callable[[str], bool]
    count_validator: Callable[[str, list[str]], bool]
    prof_validator: Callable[[str], bool]


class ProfilerChecker:
    """Unified Profiler checker supporting GPU/NPU devices"""

    TARGET_STAGES = ["actor_update", "*rollout*", "ref_*"]

    def __init__(self, device_type: str, profiler_dir: str, stages: list[str] | None = None):
        self.device_type = device_type.lower()
        self.profiler_dir = profiler_dir

        # Validate device type
        if self.device_type not in ["gpu", "npu"]:
            raise ValueError(f"Unsupported device type: {device_type}, only gpu/npu are supported")
        self.stages = stages if stages is not None else self.TARGET_STAGES
        if not self.stages:
            raise ValueError("At least one profiler stage is required")

        # Initialize device-specific configuration
        self._init_device_config()

    def _init_device_config(self):
        """Initialize validation rules for different devices (core: device differences as config)"""
        if self.device_type == "gpu":
            self.config = DeviceCheckConfig(
                search_patterns=lambda stage: [os.path.join(f"*{stage}*", "**", "*.json*"), f"*{stage}*.json*"],
                path_filter=lambda p: os.path.isfile(p) and p.endswith((".json", ".json.gz")),
                count_validator=lambda stage, paths: len(paths) > 0,
                prof_validator=lambda p: os.path.getsize(p) > 0,
            )
        else:  # NPU
            self.config = DeviceCheckConfig(
                # NPU search pattern: match ascend subdirectory under stage
                search_patterns=lambda stage: [os.path.join(stage, "*_ascend_*")],
                path_filter=os.path.isdir,
                # Each stage needs output; multiple ranks or windows may produce multiple directories.
                count_validator=lambda stage, paths: len(paths) > 0,
                # NPU: PROF_* subdirectory must exist and be a valid directory
                prof_validator=lambda d: (
                    len(glob.glob(os.path.join(d, "PROF_*"))) > 0
                    and os.path.isdir(glob.glob(os.path.join(d, "PROF_*"))[0])
                ),
            )

    def _validate_stage(self, stage: str) -> bool:
        """Match, log and validate the stage's profiler output."""
        patterns = [os.path.join(self.profiler_dir, p) for p in self.config.search_patterns(stage)]
        paths = sorted(
            {
                path
                for pattern in patterns
                for path in glob.glob(pattern, recursive=True)
                if self.config.path_filter(path)
            }
        )
        logger.info(f"[{stage}] Found {len(paths)} profiler paths (patterns: {patterns})")
        for path in paths:
            logger.info(f"[{stage}] Found: {path}")

        if not self.config.count_validator(stage, paths):
            logger.error(f"[{stage}] Unexpected profiler output count: {len(paths)}")
            return False

        for path in paths:
            if not self.config.prof_validator(path):
                logger.error(f"[{stage}] Missing or empty profiler output: {path}")
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
        for stage in self.stages:
            if not self._validate_stage(stage):
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
    parser.add_argument(
        "--stage",
        nargs="+",
        default=None,
        help=f"Stage patterns to check (default: {ProfilerChecker.TARGET_STAGES})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        checker = ProfilerChecker(device_type=args.device, profiler_dir=args.profiler_dir, stages=args.stage)
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
