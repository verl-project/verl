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
from typing import Callable

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.skip.base_skip import SKIP_REGISTRY
from verl.utils.skip.config import SkipManagerConfig


class SkipManager:
    """SkipManager is a manager for skip.

    Class attributes default here so code paths (e.g. tests or modules that only import
    ``@SkipManager.annotate_tq``) work **before** ``SkipManager.init(config)`` runs — decorators
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

    @classmethod
    def annotate_tq(cls, role: str, phase: str):
        """V1 TransferQueue-based skip decorator, unified across both phases.

        V1's split architecture separates "submit prompts" (``_add_batch_to_generate``)
        from "sample trajectories" (``ReplayBuffer.sample``).  This decorator handles
        both, selected by *phase*:

        - ``phase="submit"``: decorate ``_add_batch_to_generate``.  The method must be
          split into ``_next_train_batch`` (dataloader + uid) and ``_submit_batch_to_rollout``
          (tag registration + generate_sequences).  The decorator always calls
          ``_next_train_batch`` first (keeping the dataloader aligned even on cache-hit
          steps), then checks ``meet_precondition``.  On cache-hit it injects cached data
          into the TQ and returns; on cache-miss it calls ``_submit_batch_to_rollout``.

        - ``phase="sample"``: decorate ``ReplayBuffer.sample``.  After the original
          ``sample`` runs, if the skip instance says to save, persist the result to disk
          (phase one / cache-miss).  When ``parameter_sync_step > 1`` (separate async),
          ``sample`` is called multiple times per step; each call saves its mini-batch to
          a separate inner sub-directory, and the full set is merged on load.
        """

        def decorator(func: Callable) -> Callable:
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
                # ReplayBuffer.sample returns (batch, off_policy_metrics)
                batch = result[0] if isinstance(result, tuple) else result
                if skip_instance.should_save(step, batch.partition_id):
                    skip_instance.prepare_data(step=step, batch=batch, global_steps=step)
                return result

            return wrapper

        return decorator
