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

from dataclasses import dataclass, field

from verl.base_config import BaseConfig


@dataclass
class RolloutSkipConfig(BaseConfig):
    """Config for rollout skip behavior."""

    enable: bool = False
    dump_dir: str = "~/.verl/rollout_dump"
    steps: list[int] = field(default_factory=list)
    action: str = "cache"  # cache | repeat | random | empty, refer to SkipAction in base_skip.py

    def __post_init__(self) -> None:
        assert isinstance(self.enable, bool), f"`enable` must be bool, got {type(self.enable)}"
        assert isinstance(self.dump_dir, str), f"`dump_dir` must be str, got {type(self.dump_dir)}"
        assert isinstance(self.steps, list), f"`steps` must be list[int], got {type(self.steps)}"
        assert all(isinstance(step, int) for step in self.steps), "`steps` must contain int only"
        assert self.action in {"cache", "repeat"}, f"`action` must be one of cache/repeat, got {self.action}"


@dataclass
class AsyncRolloutSkipConfig(BaseConfig):
    """Config for rollout skip behavior."""

    enable: bool = False
    dump_dir: str = "~/.verl/rollout_dump"
    steps: list[int] = field(default_factory=list)
    action: str = "cache"  # cache | repeat | random | empty, refer to SkipAction in base_skip.py

    def __post_init__(self) -> None:
        assert isinstance(self.enable, bool), f"`enable` must be bool, got {type(self.enable)}"
        assert isinstance(self.dump_dir, str), f"`dump_dir` must be str, got {type(self.dump_dir)}"
        assert isinstance(self.steps, list), f"`steps` must be list[int], got {type(self.steps)}"
        assert all(isinstance(step, int) for step in self.steps), "`steps` must contain int only"
        assert self.action in {"cache", "repeat"}, f"`action` must be one of cache/repeat, got {self.action}"


@dataclass
class SkipManagerConfig(BaseConfig):
    """Top-level config for skip modules."""

    rollout: RolloutSkipConfig = field(default_factory=RolloutSkipConfig)
    async_rollout: AsyncRolloutSkipConfig = field(default_factory=AsyncRolloutSkipConfig)
