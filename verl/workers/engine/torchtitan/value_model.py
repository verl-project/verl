# Copyright 2026 Individual Contributor: Zupeng Wang
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

import copy
from contextlib import contextmanager

from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter


class Qwen3ValueStateDictAdapter(Qwen3StateDictAdapter):
    """Keep TorchTitan's decoder head in DCP and use HF's scalar score head."""

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)
        self.from_hf_map.pop("lm_head.weight")
        self.from_hf_map["score.weight"] = "lm_head.weight"
        self.from_hf_map["score.bias"] = "lm_head.bias"
        self._missing_initial_head = set()
        if self.fqn_to_index_mapping is not None:
            self.fqn_to_index_mapping.pop("lm_head.weight", None)
            self.fqn_to_index_mapping.setdefault("score.weight", 1)
            self.fqn_to_index_mapping.setdefault("score.bias", 1)

    @contextmanager
    def initial_hf_load(self, path):
        """Allow newly initialized head parameters while strictly loading the backbone."""
        # A causal LM checkpoint has no scalar head. Omit only that requested
        # tensor; DCP must still reject a missing backbone parameter.
        metadata = self.get_hf_storage_reader(path).read_metadata().state_dict_metadata
        head_shapes = {"score.weight": (1, self.model_config.dim), "score.bias": (1,)}
        for name, shape in head_shapes.items():
            if name in metadata and tuple(metadata[name].size) != shape:
                raise ValueError(f"Expected {name} shape {shape}, got {tuple(metadata[name].size)}")
        previous = self._missing_initial_head
        self._missing_initial_head = head_shapes.keys() - metadata.keys()
        try:
            yield
        finally:
            self._missing_initial_head = previous

    def to_hf(self, state_dict):
        """Export the scalar score head, omitting only absent initial HF parameters."""
        result = super().to_hf(state_dict)
        for name in self._missing_initial_head:
            result.pop(name, None)
        return result

    def from_hf(self, hf_state_dict):
        """Convert HF backbone and score parameters to TorchTitan names."""
        # Vocabulary logits are unrelated to token values, even when the source
        # checkpoint has an untied language-model head.
        return super().from_hf({key: value for key, value in hf_state_dict.items() if key != "lm_head.weight"})


def parallelize_qwen3_value_model(model, **kwargs):
    """Declare the scalar bias layout before TorchTitan applies parallelism."""
    shardings = model.config.lm_head.sharding_config.state_shardings
    shardings["bias"] = copy.deepcopy(shardings["weight"])
    return parallelize_qwen3(model, **kwargs)
