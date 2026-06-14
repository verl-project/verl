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

from typing import Callable

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

__all__ = ["register", "get_reward_manager_cls"]

REWARD_MANAGER: dict[str, type[RewardManagerBase]] = {}


def register(name: str) -> Callable[[type[RewardManagerBase]], type[RewardManagerBase]]:
    """Decorator to register a reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.
    """

    def decorator(cls: type[RewardManagerBase]) -> type[RewardManagerBase]:
        if name in REWARD_MANAGER and REWARD_MANAGER[name] != cls:
            raise ValueError(f"reward manager {name} has already been registered: {REWARD_MANAGER[name]} vs {cls}")
        REWARD_MANAGER[name] = cls
        return cls

    return decorator


def get_reward_manager_cls(name: str) -> type[RewardManagerBase]:
    """Get the reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.

    Returns:
        `(type)`: The reward manager class.
    """
    if name not in REWARD_MANAGER:
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER[name]
