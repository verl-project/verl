# Copyright 2025 Meituan Ltd. and/or its affiliates
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
from omegaconf import DictConfig

from verl import DataProto


def default_super_sample_func(batch: DataProto, config: DictConfig, **kwargs) -> DataProto:
    """A default super sample function that sample rollout.n trajectories."""
    return batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
