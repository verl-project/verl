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
import pickle
import time
from enum import Enum
from pathlib import Path

import numpy as np

from verl.protocol import DataProto
from verl.workers.config.rollout import RolloutConfig


def _get_skip_attr(skip_config, key: str, default):
    """Get attribute from skip config, supporting both dict and SkipConfig dataclass."""
    if isinstance(skip_config, dict):
        return skip_config.get(key, default)
    return getattr(skip_config, key, default)


class SkipAction(Enum):
    CACHE = "cache"  # cache the sample. If dump_date is found, use it. If not found, dump it.
    REPEAT = "repeat"  # Repeat the sample when gen_step reach skip.max_dump_step
    REPEAT_LAST = "repeat_last"  # Repeat the last sample when gen_step reach skip.max_dump_step


class RolloutSkip:
    """
    RolloutSkip skips sequence generation during rollout by attempting to load previously dumped data.
    If no dumped data is found, it generates new sequences and saves them to disk.

    Args:
        config: The configuration object containing rollout settings.
        rollout_wg: The worker group that handles the rollout process.

    Note:
        Whenever any of the following parameters differ from previous runs—trainer.experiment_name,
        trainer.project_name, rollout.n, or rollout.gen_batch_size—new sequences will be generated
        and saved under different filenames.


    """

    print_mark = "[RolloutSkip()] "

    def __init__(self, config, rollout_wg):
        self.rollout_config: RolloutConfig = config.actor_rollout_ref.rollout
        self.skip_config = self.rollout_config.skip
        self.is_enable = _get_skip_attr(self.skip_config, "enable", False)
        self._rollout_wg = rollout_wg

        if not self.is_enable:
            return

        self.exp_name = config.trainer.get("experiment_name", "")
        self.project_name = config.trainer.get("project_name", "")
        self.n = int(getattr(self.rollout_config, "n", 0))
        self.gbs = int(config.data.get("gen_batch_size", config.data.get("train_batch_size", 0)))
        self.response_length = config.data.get("max_response_length", 0)
        self.prompt_length = config.data.get("max_prompt_length", 0)

        self._new_batch = None
        self.curr_gen_step: int = 0  # mark the index of rollout result, start from 1
        self.curr_train_step: int = 0

        self.record_global_steps = None  # Given from xxx_ray_tainer.py, start from 1
        self.record_gen_steps = None  # Given from xxx_ray_tainer.py, start from 1
        self.__gen_offset_step = 0

        self.do_compress = _get_skip_attr(self.skip_config, "compress", True)
        self.max_dump_step = max(0, _get_skip_attr(self.skip_config, "max_dump_step", 1))  # at least dump once
        self.action = _get_skip_attr(self.skip_config, "action", SkipAction.REPEAT)
        self.action = SkipAction(self.action)

        if self.max_dump_step <= 0:
            assert self.action in [SkipAction.CACHE]

        self._create_dump_path()
        self._flag_record = False
        self.list_dumped_steps = []

    @property
    def is_activate(self) -> bool:
        """
        If RolloutSkip is enabled and the rollout worker group is set, it is considered active.
        """
        return self.is_enable and self._rollout_wg is not None

    @property
    def is_dump_step(self) -> bool:
        """
        Determine if the current step is a dump step based on the configured dump interval.
        If train_step is given, it follows the train_step, otherwise it follows the gen_step.
        """
        return self.is_activate and self.curr_train_step <= self.max_dump_step

    @property
    def num_dumped_step(self) -> int:
        return len(self.list_dumped_steps)

    def get_path_dump(self, gen_step: int = None) -> Path:
        if gen_step is None:
            gen_step = self.curr_gen_step
        return self.specify_dumped_dir.joinpath(f"genstep_{gen_step:06d}.pkl").absolute()

    def get_step_describe(self):
        return self.specify_dumped_dir.joinpath("train_step__gen_step.txt").absolute()

    def step(self):
        if self.record_global_steps is None:
            self.curr_train_step += 1
        else:
            self.curr_train_step = self.record_global_steps

        if self.record_gen_steps is None:
            self.curr_gen_step = self.curr_train_step
        else:
            self.curr_gen_step = self.record_gen_steps

    def _create_dump_path(self):
        """
        Create the directory for dumping rollout data if it doesn't exist.
        Warn if the directory is within Ray's temporary session directory.
        """

        dumped_dir = Path(_get_skip_attr(self.skip_config, "dump_dir", "/tmp/verl/rollout_dump"))
        sub_dir = (
            f"{self.exp_name}_{self.project_name}"
            + f"/GBS{self.gbs}_N{self.n}_in{self.prompt_length}_out{self.response_length}"
        )

        self.specify_dumped_dir = dumped_dir.joinpath(sub_dir)
        self.specify_dumped_dir.mkdir(parents=True, exist_ok=True)

        tmp_ray = "/tmp/ray/session"

        # Check if path is in Ray temporary directory
        if str(self.specify_dumped_dir.absolute()).startswith(tmp_ray):
            print(
                f"{self.print_mark}\033[33mWarning: \nUsing dump path ",
                f"'{self.specify_dumped_dir.absolute()}' is not recommended ",
                f"as it's located in {tmp_ray}*\033[0m",
                flush=True,
            )
        print(
            f"{self.print_mark}Rollout skip dump path set to: ",
            str(self.specify_dumped_dir.absolute()),
            flush=True,
        )

    def record(self, new_batch: DataProto, global_steps=None, gen_steps=None, *args, **kwargs):
        """Record the current training step based on the new batch.

        Args:
            new_batch (DataProto): The new batch of data being processed.
        """
        if self._rollout_wg is None:
            return
        if self._flag_record is False:
            # make sure one record only corresponds to one skip
            self._flag_record = True
            self._new_batch = new_batch
        else:
            print(
                f"{self.print_mark}Warning, duplicate record new_batch, "
                "it was not a problem if acc/reward is not cared.",
                flush=True,
            )

        if gen_steps is None:
            gen_steps = global_steps

        # Check if train_step not start from 1
        if global_steps is not None:
            if self.record_global_steps is None and global_steps > 1:
                print(f"{self.print_mark}\033[32mResume Mode.\033[0m", flush=True)
                last_train_step = global_steps - 1  # default when step file missing
                last_gen_step = 0
                try:
                    list_step = np.loadtxt(self.get_step_describe()).astype(int)
                    idx = np.where(list_step[:, 0] == (global_steps - 1))[0][-1]  # last gen_step from resume step
                    last_train_step = list_step[idx][0]
                    last_gen_step = list_step[idx][1]
                    if last_train_step + 1 != global_steps:
                        print(f"{self.print_mark}\033[31mWarning: Train step not contioues.\033[0m")
                    self.__gen_offset_step = last_gen_step
                except Exception as e:
                    print(
                        f"{self.print_mark}\033[31mFailed to read step describe file. {e.__repr__()}\033[0m",
                        flush=True,
                    )
                print(
                    f"{self.print_mark}\033[32mResume from train_step: {last_train_step}, "
                    f"gen_step: {last_gen_step}.\033[0m",
                    flush=True,
                )

        if global_steps is not None:
            self.record_global_steps = global_steps
        if gen_steps is not None:
            #! it is not right since dapo_trainer reset `gen_steps` when resume
            self.record_gen_steps = gen_steps + self.__gen_offset_step

    def wrap_generate_sequences(self):
        # if self.is_enable:
        #     self._rollout_wg = rollout_wg

        try:
            self._rollout_wg.generate_sequences = wrap_generate_sequences(self, self._rollout_wg)
            print(
                f"{self.print_mark}\033[32mSuccessfully patched `actor_rollout_wg.generate_sequences()`.\033[0m",
                flush=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"{self.print_mark}\033[31mFailed to patch `actor_rollout_wg.generate_sequences()`.\033[0m",
                flush=True,
            ) from e

    def try_load(self, step=None):
        dumped_gen_batch = None
        dumped_new_batch = None
        if step is None:
            step = self.curr_gen_step

        path_dump = self.get_path_dump(step)
        if path_dump.exists():
            try:
                # * Load
                data_dict = pickle.loads(path_dump.read_bytes())
                dataproto_decompress(data_dict)

                dumped_gen_batch = data_dict["gen_batch"]
                dumped_new_batch = data_dict["new_batch"]

                print(
                    f"{self.print_mark}\033[32mSuccessfully load pre-generated data from {path_dump}.\033[0m",
                    flush=True,
                )

                if step not in self.list_dumped_steps:
                    self.list_dumped_steps.append(step)

            except Exception:
                print(
                    f"{self.print_mark}\033[31mFailed to load pre-generated data from {path_dump}.\033[0m",
                    flush=True,
                )

        else:
            print(
                f"{self.print_mark}\033[33mNo dumped data found at gen_step {step}",
                f"from {path_dump}. The trainer will generate and dump the data for this gen_step.\033[0m",
                flush=True,
            )

        return dumped_new_batch, dumped_gen_batch

    def dump(self, outputs: DataProto):
        # todo raise error in dump is too late, fix it later.
        if self._flag_record is False or self._new_batch is None:
            raise AssertionError(
                f"{self.print_mark}\033[33mError: \n"
                + "The new_batch record is required."
                + "Please record the new_batch using `RolloutSkip.record(new_batch)` in trainer.fit().\033[0m"
            )
        self._flag_record = False

        train_step = self.record_global_steps if self.record_global_steps is not None else self.curr_train_step
        gen_step = self.record_gen_steps if self.record_gen_steps is not None else self.curr_gen_step
        data_dump = {
            "new_batch": self._new_batch,
            "gen_batch": outputs,
            "compressed": [],
            "global_steps": train_step,
            "gen_steps": gen_step,
        }

        try:
            info_compress = ""
            if self.do_compress:
                data_dump["compressed"] = ["gen_batch", "new_batch"]
                dict_info = dataproto_compress(data_dump)
                size_zip = dict_info["size_compressed_data"]
                size_data = dict_info["size_data"]
                ratio = dict_info["ratio"]

                if size_data != 0:
                    info_compress = f"{size_data / 1024**2:.3f}MB -> {size_zip / 1024**2:.3f}MB ({ratio:.1%} )"
            # Dump rollout result
            with open(str(self.get_path_dump()), "wb") as f:
                pickle.dump(data_dump, f)
            # Dump info of train_step and gen_step for resume
            with open(str(self.get_step_describe()), "a") as f:
                f.write(f"{train_step} {gen_step}\n")

            print(
                f"{self.print_mark}\033[32mSuccessfully dump data in {self.get_path_dump()}\033[0m",
                info_compress,
                flush=True,
            )
            if self.curr_gen_step not in self.list_dumped_steps:
                self.list_dumped_steps.append(self.curr_gen_step)

        except Exception as e:
            print(
                f"{self.print_mark}\033[31mFailed to dump data in {self.get_path_dump()}: {e}\033[0m",
                flush=True,
            )

    def replace_curr_new_batch(self, dumped_new_batch: DataProto):
        """Replace the current new_batch's content with that from the dumped_new_batch.
        In case of [Answer] mismatch.
        """

        if self._flag_record is False:
            raise AssertionError(
                f"{self.print_mark}\033[33mError: \n"
                + "The new_batch is not recorded. Please record the new_batch"
                + "using `RolloutSkip.record(new_batch)`. \033[0m"
            )
        self._flag_record = False

        self._new_batch.batch = dumped_new_batch.batch
        self._new_batch.non_tensor_batch = dumped_new_batch.non_tensor_batch
        self._new_batch.meta_info = dumped_new_batch.meta_info


def wrap_generate_sequences(rolloutskip: RolloutSkip, rollout_wg):
    generate_sequences = rollout_wg.generate_sequences

    def rollout_skip_wrap_fn(batch, **kwargs) -> DataProto:
        rolloutskip.step()
        # Record input batch as new_batch so dump() / replace_curr_new_batch() have it
        rolloutskip.record(batch)
        return_batch = None

        if rolloutskip.is_dump_step:
            # * try load
            dumped_new_batch, return_batch = rolloutskip.try_load()

            if return_batch is None:
                # 1. Generation
                return_batch = generate_sequences(batch, **kwargs)
                # 2. Dump
                rolloutskip.dump(return_batch)
            else:
                rolloutskip.replace_curr_new_batch(dumped_new_batch)

        elif rolloutskip.action == SkipAction.CACHE:
            return_batch = generate_sequences(batch, **kwargs)

        elif rolloutskip.action == SkipAction.REPEAT:
            if rolloutskip.num_dumped_step == 0:
                return_batch = generate_sequences(batch, **kwargs)
                rolloutskip.dump(return_batch)
            else:
                target_step = rolloutskip.list_dumped_steps[
                    (rolloutskip.curr_gen_step - 1) % rolloutskip.num_dumped_step
                ]
                dumped_new_batch, return_batch = rolloutskip.try_load(step=target_step)
                if return_batch is None:
                    return_batch = generate_sequences(batch, **kwargs)
                    rolloutskip.dump(return_batch)
                else:
                    rolloutskip.replace_curr_new_batch(dumped_new_batch)

        elif rolloutskip.action == SkipAction.REPEAT_LAST:
            target_step = rolloutskip.list_dumped_steps[-1]
            dumped_new_batch, return_batch = rolloutskip.try_load(step=target_step)
            if return_batch is None:
                return_batch = generate_sequences(batch, **kwargs)
                rolloutskip.dump(return_batch)
            else:
                rolloutskip.replace_curr_new_batch(dumped_new_batch)

            # clean
        return return_batch

    return rollout_skip_wrap_fn


def dataproto_compress(dict_data: dict) -> dict[str, DataProto]:
    try:
        import pyzstd

        compresser = pyzstd
    except ImportError:
        import zlib

        compresser = zlib

    dict_data["compresser_name"] = compresser.__name__

    key_compress = dict_data.get("compressed", [])

    size_data = 0
    size_compressed_data = 0

    print("Compress dumped data...", flush=True)
    time_pickle = 0
    time_compress = 0
    for key in key_compress:
        time_start = time.time()
        _data = pickle.dumps(dict_data[key])
        time_pickle += time.time() - time_start
        size_data += len(_data)

        time_start = time.time()
        compressed_data = compresser.compress(_data)
        time_compress += time.time() - time_start

        size_compressed_data += len(compressed_data)

        dict_data[key] = compressed_data

    dict_info = {
        "size_compressed_data": size_compressed_data,
        "size_data": size_data,
        "time_pickle": time_pickle,
        "time_compress": time_compress,
        "ratio": size_compressed_data / size_data if size_data != 0 else None,
    }

    return dict_info


def dataproto_decompress(dict_data: dict[str, DataProto]) -> dict[str, DataProto]:
    key_compresser_name = dict_data.get("compresser_name", "zlib")
    if key_compresser_name == "zlib":
        import zlib

        compresser = zlib
    elif key_compresser_name == "pyzstd":
        import pyzstd

        compresser = pyzstd

    key_compress = dict_data.get("compressed", [])

    for key in key_compress:
        compressed_data = compresser.decompress(dict_data[key])
        _data = pickle.loads(compressed_data)
        dict_data[key] = _data

    dict_data["compressed"] = []


def read_dumped_data(path_dump: Path) -> dict[str, DataProto]:
    """
    Common function to read and decompress dumped data from a specified path.

    ```
    import verl
    from verl.utils.rollout_skip import read_dumped_data

    dumped_data = read_dumped_data("tmp/rollout_dump/DAPO-Qwen2.5-0.5B_DAPO/GBS4_N4_in2048_out4096/genstep_000001.pkl")

    print(dumped_data["new_batch"])
    print(dumped_data["gen_batch"])
    ```

    """
    path_dump = Path(path_dump)
    if path_dump.is_file():
        with open(path_dump, "rb") as f:
            data_dump = pickle.load(f)
    else:
        raise FileNotFoundError(f"File {path_dump} does not exist.")

    dataproto_decompress(data_dump)

    return data_dump