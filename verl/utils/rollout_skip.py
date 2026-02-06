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
import logging
import os
from pathlib import Path

from verl.protocol import DataProto

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class RolloutSkip:
    """
    RolloutSkip skips sequence generation during rollout by attempting to load previously dumped data.
    If no dumped data is found, it generates new sequences and saves them to disk.

    Args:
        config: The configuration object containing rollout settings.
        rollout_wg: The worker group that handles the rollout process.

    Note:
        When rollout.n or rollout.gen_batch_size differ from previous runs,
        new sequences will be generated and saved with different filenames.
    """

    print_mark = "[RolloutSkip()]"

    def __init__(self, config, rollout_wg):
        self.rollout_config = config.actor_rollout_ref.rollout
        self.exp_name = config.data.get("experiment_name", "")
        self.project_name = config.data.get("project_name", "")

        self.n = int(self.rollout_config.get("n", 0))
        self.gbs = int(config.data.get("gen_batch_size", config.data.get("train_batch_size", 0)))

        self.dumped_dir = Path(self.rollout_config.get("skip_dump_dir", "/tmp/verl/rollout_dump"))
        self.dumped_dir.mkdir(parents=True, exist_ok=True)

        # Check if path is in Ray temporary directory
        if str(self.dumped_dir.absolute()).startswith("/tmp/ray/session"):
            logger.warning(
                f"{self.print_mark} Using dump path "
                f"'{self.dumped_dir.absolute()}' is not recommended "
                "as it's located in /tmp/ray/session*"
            )

        logger.info(f"{self.print_mark} Rollout skip dump path set to: {self.dumped_dir.absolute()}")

        self._rollout_wg = rollout_wg

    @property
    def curr_path_dump(self):
        return self.dumped_dir.joinpath(f"{self.exp_name}_{self.project_name}_GBS{self.gbs}__N{self.n}").absolute()

    def wrap_generate_sequences(self):
        try:
            self._rollout_wg.generate_sequences = wrap_generate_sequences(self, self._rollout_wg)
            logger.info(f"{self.print_mark} Successfully patched `actor_rollout_wg.generate_sequences()`")
        except Exception as e:
            raise RuntimeError(f"{self.print_mark} Failed to patch `actor_rollout_wg.generate_sequences()`") from e

    def try_load(self):
        if not self.curr_path_dump.exists():
            logger.info(
                f"{self.print_mark} No data dump found at {self.curr_path_dump}. The trainer will generate and"
                " automatically dump the data for this first run."
            )
            return None

        try:
            # * Load
            ret_batch = DataProto.load_from_disk(self.curr_path_dump)
            logger.info(f"{self.print_mark} Successfully load pre-generated data from {self.curr_path_dump}")
            return ret_batch
        except Exception as e:
            logger.error(f"{self.print_mark} Failed to load pre-generated data from {self.curr_path_dump}. Error: {e}")
            return None

    def dump(self, outputs: DataProto):
        try:
            outputs.save_to_disk(self.curr_path_dump)
            logger.info(f"{self.print_mark} Successfully dump data in {self.curr_path_dump}")
        except Exception as e:
            logger.error(f"{self.print_mark} Failed to dump data in {self.curr_path_dump}: {e}")


def wrap_generate_sequences(rolloutskip: RolloutSkip, rollout_wg):
    generate_sequences = rollout_wg.generate_sequences

    def warp_fn(batch, **kwargs):
        gen_batch_output = rolloutskip.try_load()

        if gen_batch_output is None:
            # * 1. Generation
            gen_batch_output = generate_sequences(batch, **kwargs)
            # * 2. Dump
            rolloutskip.dump(gen_batch_output)
        return gen_batch_output

    return warp_fn
