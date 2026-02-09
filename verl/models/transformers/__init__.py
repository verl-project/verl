# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.models.transformers.tiled_mlp import apply_tiled_mlp_monkey_patch

__all__ = [
    "apply_monkey_patch",
    "apply_tiled_mlp_monkey_patch",
]
