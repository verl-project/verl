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
"""Configurable driver-side hooks that run immediately before advantage computation."""

from typing import Any

from verl import DataProto
from verl.utils.import_utils import load_class_from_fqn


class PreAdvantageHook:
    """Base class for driver-side batch transforms before advantage computation.

    Hooks receive the full trainer config at construction time and may transform
    the in-memory :class:`DataProto` or emit scalar metrics. Exceptions propagate
    so a partially applied algorithm transform cannot be ignored.
    """

    def __init__(self, config: Any, hook_config: Any):
        self.config = config
        self.hook_config = hook_config

    def __call__(self, data: DataProto, **kwargs) -> tuple[DataProto, dict[str, float]]:
        """Transform ``data`` and return metrics; the default hook is a no-op."""
        return data, {}


def build_pre_advantage_hooks(config: Any) -> list[PreAdvantageHook]:
    """Build enabled hooks from ``trainer.pre_advantage_hooks`` in declaration order."""
    hook_configs = config.trainer.get("pre_advantage_hooks", {})
    hooks: list[PreAdvantageHook] = []
    for name, hook_config in hook_configs.items():
        if not hook_config.get("enable", True):
            continue
        hook_class_fqn = hook_config.get("hook_class")
        if not hook_class_fqn:
            raise ValueError(f"Enabled pre-advantage hook {name!r} must define hook_class")
        hook_class = load_class_from_fqn(hook_class_fqn, f"pre-advantage hook {name!r}")
        if not issubclass(hook_class, PreAdvantageHook):
            raise TypeError(f"Pre-advantage hook {name!r} must subclass PreAdvantageHook")
        hooks.append(hook_class(config=config, hook_config=hook_config))
    return hooks


def run_pre_advantage_hooks(
    hooks: list[PreAdvantageHook], data: DataProto, *, trainer: Any
) -> tuple[DataProto, dict[str, float]]:
    """Run hooks sequentially so each hook observes the previous hook's output."""
    metrics: dict[str, float] = {}
    for hook in hooks:
        data, hook_metrics = hook(data, trainer=trainer)
        metrics.update(hook_metrics)
    return data, metrics
