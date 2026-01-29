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
"""
Speculator adapter interface and factory.
This keeps trainers decoupled from concrete speculator implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from verl.utils.import_utils import load_class_from_fqn, load_extern_object
from verl.utils.fsdp_utils import fully_shard, maybe_patch_fsdp_module
import torch


class SpeculatorAdapter(ABC):
    device_mesh: Any
    speculator: Any

    @abstractmethod
    def build_speculator_module(self, model):
        """Build and optionally attach a speculator module to the model."""

    @abstractmethod
    def get_optimizer_params(self):
        """Return the parameters to optimize."""

    @abstractmethod
    def compute_speculator_loss(
        self,
        fsdp_model,
        input_ids,
        attention_mask=None,
        position_ids=None,
        loss_mask=None,
        hidden_states=None,
        spec_logits=None,
        packed_seq_params=None,
    ):
        """Compute speculator loss."""

    def apply_fsdp2_speculator(self, fsdp_model, fsdp_kwargs: Optional[dict]):
        assert self.speculator is not None, "Speculator module is not built yet."
        speculator_module = self.speculator
        with maybe_patch_fsdp_module(speculator_module):
            speculator_module = fully_shard(speculator_module, **fsdp_kwargs)
        # fsdp_model.speculator =speculator_module
        return speculator_module

    def _get_speculator_module(self):
        return getattr(self, "speculator", None)

    def _get_speculator_config_obj(self, speculator_module):
        if speculator_module is None:
            return None
        return getattr(speculator_module, "config", None)

    def _maybe_pad_nested(self, tensor, padding):
        if isinstance(tensor, torch.Tensor) and tensor.is_nested:
            return torch.nested.to_padded_tensor(tensor, padding=padding)
        return tensor

    def _maybe_normalize_hidden_layout(self, hidden_states, attention_mask, input_ids):
        if hidden_states is None or not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 3:
            return hidden_states
        batch_size = None
        seq_len = None
        if attention_mask is not None and attention_mask.dim() >= 2:
            batch_size, seq_len = attention_mask.shape[:2]
        elif input_ids is not None and isinstance(input_ids, torch.Tensor) and input_ids.dim() >= 2:
            batch_size, seq_len = input_ids.shape[:2]
        if batch_size is None or seq_len is None:
            return hidden_states
        if hidden_states.size(1) == batch_size and hidden_states.size(0) != batch_size:
            return hidden_states.transpose(0, 1).contiguous()
        return hidden_states

    def _maybe_unpack_packed_hidden(self, input_ids, attention_mask, hidden_states, packed_seq_params):
        if packed_seq_params is None or hidden_states is None:
            return hidden_states
        if isinstance(hidden_states, torch.Tensor) and hidden_states.is_nested:
            return hidden_states
        if attention_mask is None:
            if isinstance(input_ids, torch.Tensor) and input_ids.is_nested:
                offsets = input_ids.offsets().diff().tolist()
                seq_len = max(offsets) if offsets else 0
                attention_mask = torch.zeros(
                    (len(offsets), seq_len), dtype=torch.bool, device=input_ids.device
                )
                for i, seqlen in enumerate(offsets):
                    attention_mask[i, :seqlen] = True
            elif isinstance(input_ids, torch.Tensor) and input_ids.dim() >= 2:
                attention_mask = torch.ones(
                    input_ids.shape[:2], dtype=torch.bool, device=input_ids.device
                )
            else:
                return hidden_states
        attention_mask = self._maybe_pad_nested(attention_mask, padding=0)
        if attention_mask is None:
            return hidden_states
        if hidden_states.dim() >= 2 and hidden_states.size(0) == attention_mask.size(0) and (
            hidden_states.size(1) == attention_mask.size(1)
        ):
            return hidden_states
        packed_states = hidden_states
        if hidden_states.dim() == 2:
            packed_states = hidden_states.unsqueeze(0)
        elif hidden_states.dim() == 3 and hidden_states.size(1) == 1:
            packed_states = hidden_states.transpose(0, 1)
        if packed_states.dim() >= 2 and packed_states.size(0) == 1:
            from verl.models.mcore.util import postprocess_packed_seqs

            batch_size, seq_len = attention_mask.shape[:2]
            unpacked = postprocess_packed_seqs(packed_states, packed_seq_params, attention_mask, batch_size, seq_len)
            return unpacked
        return hidden_states

    def _slice_speculator_inputs(self, input_ids, hidden_states, n_predict):
        # Alignment rule: hidden_states[t] should predict tokens starting at input_ids[t+1],
        # and we need n_predict extra tokens to the right, so drop the last n_predict+1 states.
        hidden = hidden_states[:, : -(n_predict + 1), :]
        seq_ids = input_ids[:, 1:]
        return hidden, seq_ids

    def get_optimizer_params(self):
        speculator_module = self._get_speculator_module()
        if speculator_module is not None:
            return speculator_module.parameters()
        return None


def _load_custom_adapter(speculator_adapter_config: Any):
    if isinstance(speculator_adapter_config, str):
        return load_class_from_fqn(speculator_adapter_config, description="speculator adapter")

    if hasattr(speculator_adapter_config, "get"):
        fqn = speculator_adapter_config.get("fqn", None)
        if fqn:
            return load_class_from_fqn(fqn, description="speculator adapter")
        path = speculator_adapter_config.get("path", None)
        name = speculator_adapter_config.get("name", None)
        if path and name:
            return load_extern_object(path, name)

    return None


def build_speculator_adapter(
    config,
    model_config,
    device_name,
    device_mesh,
    torch_dtype,
):
    speculator_adapter_config = None
    if config is not None and hasattr(config, "model"):
        speculator_adapter_config = getattr(config.model, "speculator_adapter", None)
    if speculator_adapter_config is None and model_config is not None:
        speculator_adapter_config = getattr(model_config, "speculator_adapter", None)

    adapter_cls = _load_custom_adapter(speculator_adapter_config)
    if adapter_cls is None:
        raise ValueError(
            "speculator_adapter is not configured. Please set config.model.speculator_adapter "
            "to a valid adapter class or module path."
        )

    return adapter_cls(
        config=config,
        model_config=model_config,
        device_name=device_name,
        device_mesh=device_mesh,
        torch_dtype=torch_dtype,
    )
