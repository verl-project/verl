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
        if prompts is None and len(args) > 1:
            return cls._should_bypass_for_validation_tensordict(args, kwargs)
        if prompts is None:
            return False
        if hasattr(prompts, "non_tensor_data"):
            return cls._should_bypass_for_validation_tensordict(args, kwargs)
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

    @classmethod
    def annotate_tq(cls, role: str, phase: str):
        """V1 TransferQueue-based skip decorator.

        The synchronous ``phase="submit"`` branch supports inject-then-sample
        callers. Streaming dataloader actors use the asynchronous ``phase="sample"``
        branch:

        - Cache misses run the original ``sample`` and persist its ``KVBatchMeta``.
        - Cache hits advance the actor-owned dataloader, inject one cached inner
          batch under fresh uids, and return that exact batch without normal sampling.
        """

        def decorator(func: Callable) -> Callable:
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(self, *args, **kwargs):
                    if phase != "sample":
                        return await func(self, *args, **kwargs)
                    skip_instance = cls.skip_instances.get(role)
                    if skip_instance is None or not skip_instance.is_enabled():
                        return await func(self, *args, **kwargs)

                    call = inspect.signature(func).bind(self, *args, **kwargs)
                    call.apply_defaults()
                    step = call.arguments["global_steps"]
                    partition_id = call.arguments["partition_id"]
                    prompts = call.arguments["prompts"]
                    if partition_id != "train" or prompts is not None or step not in skip_instance.steps:
                        return await func(self, *args, **kwargs)

                    if not skip_instance.should_load_v1_sample(step, self.trainer_mode):
                        result = await func(self, *args, **kwargs)
                        batch = result[0]
                        if skip_instance.should_save(step, batch.partition_id):
                            skip_instance.prepare_data(step=step, batch=batch, global_steps=step)
                        return result

                    cls.set_step(step)
                    self.global_steps = step
                    if hasattr(self, "_validation_active"):
                        self._validation_active = False
                    fresh_batch = self._next_train_batch(call.arguments["batch_size"])
                    batch = skip_instance.load_v1_sample(
                        step=step,
                        new_prompt_uids=list(fresh_batch["uid"]),
                        global_steps=step,
                        partition_id=partition_id,
                    )
                    return batch, {}, len(batch)

                return async_wrapper

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                skip_instance = cls.skip_instances.get(role)
                if skip_instance is None or not skip_instance.is_enabled():
                    return func(self, *args, **kwargs)
                step = cls.step
                if step not in skip_instance.steps:
                    return func(self, *args, **kwargs)

                if phase == "submit":
                    # Always prepare batch (keeps dataloader aligned on cache-hit steps)
                    batch = self._next_train_batch()
                    if skip_instance.maybe_load_and_inject(step, list(batch["uid"])):
                        return
                    self._submit_batch_to_rollout(batch)
                    return

                # phase == "sample": post-save after original sample runs
                result = func(self, *args, **kwargs)
                # Sampling returns (batch, off_policy_metrics)
                batch = result[0] if isinstance(result, tuple) else result
                if skip_instance.should_save(step, batch.partition_id):
                    skip_instance.prepare_data(step=step, batch=batch, global_steps=step)
                return result

            return wrapper

        return decorator

    @staticmethod
    def _should_bypass_for_validation_tensordict(args, kwargs) -> bool:
        """Check ``validate`` flag in TensorDict's non_tensor_data for V1 path."""
        prompts = kwargs.get("prompts", None)
        if prompts is None and len(args) > 1:
            prompts = args[1]
        if prompts is None:
            return False
        try:
            return bool(getattr(prompts, "non_tensor_data", {}).get("validate", False))
        except Exception:
            return False
