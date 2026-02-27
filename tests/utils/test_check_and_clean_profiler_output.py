import argparse
import glob
import logging
import os
import shutil
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


def clean_profiler_output(profiler_dir: str):
    """Clean up profiler deliverables"""
    logger.info("Starting cleanup...")
    if os.path.exists(profiler_dir):
        try:
            shutil.rmtree(profiler_dir)
            logger.info(f"Successfully deleted directory: {profiler_dir}")
        except Exception as e:
            logger.error(f"Failed to delete directory: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check and clean Profiler deliverables")
    parser.add_argument("--profiler-dir", type=str, default="./profiler_data", help="Path to profiler data directory")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if check_profiler_output(args.profiler_dir):
            sys.exit(0)
        else:
            sys.exit(1)
    finally:
        clean_profiler_output(args.profiler_dir)


if __name__ == "__main__":
    main()
