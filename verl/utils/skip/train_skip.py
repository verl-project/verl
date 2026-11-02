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

from verl.protocol import DataProto
from verl.utils.skip.base_skip import BaseSkip, SkipAction, register_skip


@register_skip("train")
class TrainSkip(BaseSkip):
    """Bypass training stages, preserving the batch and all surrounding lifecycle hooks."""

    support_actions = [SkipAction.EMPTY]

    def __init__(self, local_config, global_config):
        super().__init__(local_config, global_config)
        self.scope = local_config.scope

    def is_active(self, step: int, *, scope: str | None = None, **kwargs) -> bool:
        """A full compute skip also covers updates; steps start at one in the trainer."""
        return (
            step >= 1
            and super().is_active(step)
            and (scope is None or scope == self.scope or (self.scope == "train_compute" and scope == "updates"))
        )

    def meet_precondition(self, step, func, trainer, batch, *args, **kwargs) -> bool:
        """Support V1 queue metadata and V0/fully-async DataProto training batches."""
        if isinstance(batch, DataProto):
            return not batch.meta_info.get("validate", False)
        return getattr(batch, "partition_id", None) == "train"

    def warp_function(self, step, func, trainer, batch, *args, **kwargs):
        """Return the existing batch without training, modifying queue data, or disk I/O."""
        return batch

    def prepare_data(self, step, result, *args, **kwargs):
        """Training skip does not persist or replay results."""
