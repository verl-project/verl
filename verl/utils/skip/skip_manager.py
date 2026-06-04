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

import functools
import inspect
from typing import Callable

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.skip.base_skip import SKIP_REGISTRY
from verl.utils.skip.config import SkipManagerConfig


class SkipManager:
    """SkipManager is a manager for skip.

    Class attributes default here so code paths (e.g. tests or modules that only import
    ``@SkipManager.annotate``) work **before** ``SkipManager.init(config)`` runs — decorators
    then no-op until the trainer initializes skip state.
    """

    config: SkipManagerConfig | None = None
    step: int = -1
    # This step is shared across all skip_instances
    # Different enabled skip_instances (no online step acquisition) shall use the same step definition.
    skip_instances: dict = {}  # noqa: RUF012 — intentionally mutable class defaults, reset in ``init``

    @classmethod
    def init(cls, config):
        cls.config = omega_conf_to_dataclass(config.skip, dataclass_type=SkipManagerConfig)
        cls.step = -1
        cls.skip_instances = {}
        for name, skip_cls in SKIP_REGISTRY.items():
            local_cfg = getattr(cls.config, name, None)
            if local_cfg is None:
                continue
            instance = skip_cls(local_cfg, config)
            cls.skip_instances[name] = instance

    @classmethod
    def set_step(cls, step: int):
        cls.step = step

    @staticmethod
    def _get_prompts_batch(args, kwargs):
        """Resolve ``DataProto`` from decorated ``generate_sequences(self, prompts, ...)`` calls."""
        prompts = kwargs.get("prompts")
        if prompts is not None:
            return prompts
        if len(args) > 1:
            return args[1]
        if len(args) == 1 and hasattr(args[0], "meta_info"):
            return args[0]
        return None

    @classmethod
    def _should_bypass_for_validation(cls, args, kwargs) -> bool:
        prompts = cls._get_prompts_batch(args, kwargs)
        if prompts is None:
            return False
        meta_info = getattr(prompts, "meta_info", None) or {}
        return bool(meta_info.get("validate", False))

    @classmethod
    def annotate(cls, role: str, **kwargs_outer) -> Callable:
        def decorator(func: Callable) -> Callable:
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs_inner):
                    if cls._should_bypass_for_validation(args, kwargs_inner):
                        return await func(*args, **kwargs_inner)
                    skip_instance = cls.skip_instances.get(role)
                    if skip_instance is None or not skip_instance.is_enabled():
                        return await func(*args, **kwargs_inner)
                    if skip_instance.support_online_step:
                        step = skip_instance.extract_step(*args, **kwargs_inner)
                    else:
                        step = cls.step
                    if step not in skip_instance.steps:
                        return await func(*args, **kwargs_inner)
                    if skip_instance.meet_precondition(step, func, *args, **kwargs_inner):
                        return skip_instance.warp_function(step, func, *args, **kwargs_inner)
                    result = await func(*args, **kwargs_inner)
                    skip_instance.prepare_data(step, result, *args, **kwargs_inner)
                    return result

                return async_wrapper

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs_inner):
                if cls._should_bypass_for_validation(args, kwargs_inner):
                    return func(*args, **kwargs_inner)
                skip_instance = cls.skip_instances.get(role)
                if skip_instance is None or not skip_instance.is_enabled():
                    return func(*args, **kwargs_inner)
                if skip_instance.support_online_step:
                    step = skip_instance.extract_step(*args, **kwargs_inner)
                else:
                    step = cls.step
                if step not in skip_instance.steps:
                    return func(*args, **kwargs_inner)
                if skip_instance.meet_precondition(step, func, *args, **kwargs_inner):
                    return skip_instance.warp_function(step, func, *args, **kwargs_inner)
                result = func(*args, **kwargs_inner)
                skip_instance.prepare_data(step, result, *args, **kwargs_inner)
                return result

            return sync_wrapper

        return decorator
