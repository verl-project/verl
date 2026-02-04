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

from transformers import PretrainedConfig


class PI0TorchConfig(PretrainedConfig):
    model_type = "pi0_torch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.state_norm_stats = kwargs.get("state_norm_stats", {})
        self.action_norm_stats = kwargs.get("action_norm_stats", {})
        self.pi05_enabled = kwargs.get("pi05_enabled", False)
