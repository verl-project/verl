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
import json
from pathlib import Path
from typing import Callable

from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.utils.skip.base_skip import BaseSkip, SkipAction, register_skip


@register_skip("rollout")
class RolloutSkip(BaseSkip):
    """RolloutSkip skips sequence generation during rollout by attempting to load previously dumped data."""

    support_actions = [SkipAction.CACHE, SkipAction.REPEAT]
    print_mark = "[RolloutSkip()] "
    gen_batch_name = "gen_batch.dp"
    meta_name = "meta.json"

    def __init__(self, local_config, global_config):
        super().__init__(local_config, global_config)
        # prepare experiment info
        self.exp_name = global_config.trainer.get("experiment_name", "default_experiment_name")
        self.project_name = global_config.trainer.get("project_name", "default_project_name")
        self.n = int(OmegaConf.select(global_config, "actor_rollout_ref.rollout.n", default=0))
        self.gbs = int(
            OmegaConf.select(
                global_config,
                "data.gen_batch_size",
                default=OmegaConf.select(global_config, "data.train_batch_size", default=0),
            )
        )
        self.response_length = OmegaConf.select(global_config, "data.max_response_length", default=0)
        self.prompt_length = OmegaConf.select(global_config, "data.max_prompt_length", default=0)

    def meet_precondition(self, step: int, func: Callable, *args, **kwargs) -> bool:
        if self.action == SkipAction.CACHE:
            if not self._check_valid_step_path(self._get_step_dump_dir(step)):
                print(
                    f"{self.print_mark}\033[33mNo dumped data found at step {step} "
                    f"from {self._get_project_dump_dir()}. "
                    f"The trainer will generate and dump the data for this step.\033[0m",
                    flush=True,
                )
                return False
            else:
                return True

        elif self.action == SkipAction.REPEAT:
            if self._find_latest_step(step) == -1:
                print(
                    f"{self.print_mark}\033[33mNo dumped data found "
                    f"from {self._get_project_dump_dir()}. "
                    f"The trainer will generate and dump the data.\033[0m",
                    flush=True,
                )
                return False
            return True
        return False

    def warp_function(self, step: int, func: Callable, *args, **kwargs):
        """Load cached gen batch; ``*args``/``kwargs`` mirror the decorated call (e.g. ``self, prompts``)."""
        if self.action == SkipAction.CACHE:
            load_step = step
        elif self.action == SkipAction.REPEAT:
            load_step = self._find_latest_step(step)
            if load_step == -1:
                raise RuntimeError(
                    f"{self.print_mark}repeat action expected dumped data for step {step}, "
                    f"but none was found under {self._get_project_dump_dir()}"
                )
        else:
            load_step = step
        step_dir = self._get_step_dump_dir(load_step)
        gen_batch_path = step_dir.joinpath(self.gen_batch_name)
        result = DataProto.load_from_disk(gen_batch_path)
        print(
            f"{self.print_mark}\033[33mLoad generate result at step {load_step} "
            f"(request step {step}) from {gen_batch_path}\033[0m",
            flush=True,
        )
        return result

    def prepare_data(self, step: int, result, *args, **kwargs):
        step_dir = self._get_step_dump_dir(step)
        try:
            step_dir.mkdir(parents=True, exist_ok=True)
            result.save_to_disk(step_dir.joinpath(self.gen_batch_name))
            meta_path = step_dir.joinpath(self.meta_name)
            meta_path.write_text(json.dumps({"global_steps": step}))
            print(
                f"{self.print_mark}\033[33mDump generate result at step {step} to {step_dir}\033[0m",
                flush=True,
            )
        except Exception as e:
            print(
                f"{self.print_mark}\033[31mFailed to dump generate result at step {step} to {step_dir}: {e}\033[0m",
                flush=True,
            )

    def _get_project_dump_dir(self) -> Path:
        dumped_dir = Path(self.dump_dir).expanduser().resolve()
        sub_dir = (
            f"{self.exp_name}_{self.project_name}"
            + f"/GBS{self.gbs}_N{self.n}_in{self.prompt_length}_out{self.response_length}"
        )
        dumped_dir = dumped_dir.joinpath(sub_dir).absolute()
        return dumped_dir

    def _get_step_dump_dir(self, step) -> Path:
        return self._get_project_dump_dir().joinpath(f"{step}").absolute()

    def _check_valid_step_path(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        gen_batch_path = path.joinpath(self.gen_batch_name)
        meta_path = path.joinpath(self.meta_name)
        return gen_batch_path.exists() and gen_batch_path.is_file() and meta_path.exists() and meta_path.is_file()

    def _get_available_steps(self) -> list[int]:
        result: list[int] = []
        project_dir = self._get_project_dump_dir()
        if not project_dir.is_dir():
            return result
        for child in project_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                step = int(child.name)
            except ValueError:
                continue
            if not self._check_valid_step_path(child):
                continue
            result.append(step)
        return sorted(result)

    def _find_latest_step(self, step: int) -> int:
        """Prefer exact ready step, else max step < current, else min step > current; -1 if none."""
        if self._check_valid_step_path(self._get_step_dump_dir(step)):
            return step
        available = self._get_available_steps()
        if not available:
            return -1
        # try to find the closest step
        smaller_steps = [this_step for this_step in available if this_step < step]
        if smaller_steps:
            return smaller_steps[-1]
        larger_steps = [this_step for this_step in available if this_step > step]
        if larger_steps:
            return larger_steps[0]
        return -1


def parse_async_rollout_sample_step(sample_id: str) -> int:
    """Parse the prompt **feed index** embedded in ``uid_sample_{epoch}_{index}`` or ``sample_{epoch}_{index}``.

    The trailing integer is Rollouter ``global_steps`` at feed time: monotonic order in which
    prompts are submitted to the async pipeline. It is **not** trainer ``global_steps``, parameter
    sync version, or guaranteed completion order under concurrent rollout.
    """
    # Strip optional "uid_" prefix (set in non_tensor_batch["uid"])
    if sample_id.startswith("uid_"):
        sample_id = sample_id[4:]
    parts = sample_id.split("_")
    if len(parts) != 3 or parts[0] != "sample":
        raise ValueError(f"Invalid async rollout sample_id: {sample_id!r}, expected sample_<epoch>_<feed_index>")
    return int(parts[-1])


@register_skip("async_rollout")
class AsyncRolloutSkip(RolloutSkip):
    """Rollout skip for fully async policy (``skip.async_rollout``)."""

    support_online_step = True

    def extract_step(self, *args, **kwargs) -> int:
        # generate_sequences_single(self, prompts)
        # sample_id is embedded in prompts.non_tensor_batch["uid"]
        prompts = args[1] if len(args) > 1 else kwargs.get("prompts")
        if prompts is None:
            raise ValueError("async_rollout extract_step expects prompts as the second argument")
        uid_array = prompts.non_tensor_batch.get("uid")
        if uid_array is None or len(uid_array) == 0:
            raise ValueError("async_rollout extract_step expects uid in prompts.non_tensor_batch")
        return parse_async_rollout_sample_step(str(uid_array[0]))
