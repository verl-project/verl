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
import random

from omegaconf import DictConfig

from verl import DataProto


def random_filter(batch: DataProto, config: DictConfig, **kwargs) -> DataProto:
    """Randomly truncates the input batch.

    Generates a random integer k between 0 and len(batch), and keeps
    only the first k elements (prefix) of the batch.

    Args:
        batch (DataProto): The input data batch.
        **kwargs: Arbitrary keyword arguments (unused).

    Returns:
        DataProto: The truncated batch.
    """
    random_number = random.randint(0, len(batch))
    batch = batch[:random_number]
    return batch
