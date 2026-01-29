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

import torch
from torch import nn

from verl.models.speculator.mlp_speculator import MLPSpeculator, MLPSpeculatorConfig
from verl.trainer.speculators.interface import SpeculatorAdapter

import os
class MLPSpeculatorAdapter(SpeculatorAdapter):
    def __init__(
        self,
        config,
        model_config,
        device_name,
        device_mesh,
        torch_dtype,
    ):
        self.config = config
        self.model_config = model_config
        self.device_name = device_name
        self.device_mesh = device_mesh
        self.torch_dtype = torch_dtype

        speculator_config = None
        if self.config is not None and hasattr(self.config, "model"):
            speculator_config = getattr(self.config.model, "speculator", None)
        if speculator_config is None:
            speculator_config = getattr(self.model_config, "speculator", None)
        self.speculator_config = speculator_config

        self.speculator = None

    def build_speculator_module(self, model):
        if self.speculator_config is None:
            return None

        hf_config = self.model_config.hf_config if hasattr(self.model_config, "hf_config") else self.model_config
        hidden_size = hf_config.hidden_size
        vocab_size = hf_config.vocab_size

        speculator_config_dict = {
            "n_predict": self.speculator_config.get("n_predict", 5),
            "emb_dim": self.speculator_config.get("emb_dim", hidden_size),
            "inner_dim": self.speculator_config.get("inner_dim", hidden_size),
            "vocab_size": vocab_size,
            "tie_weights": self.speculator_config.get("tie_weights", False),
            "scale_input": self.speculator_config.get("scale_input", False),
        }

        base_model_name_or_path = None
        if self.config is not None and hasattr(self.config, "model"):
            base_model_name_or_path = getattr(self.config.model, "path", None)
        if base_model_name_or_path is None:
            base_model_name_or_path = getattr(self.model_config, "local_path", None)
        if base_model_name_or_path is None:
            base_model_name_or_path = "unknown"

        speculator_config_dict["base_model_name_or_path"] = base_model_name_or_path
        config_obj = MLPSpeculatorConfig(**speculator_config_dict)
        self.speculator = MLPSpeculator(config_obj)


        for param in model.parameters():
            param.requires_grad = False
        for param in self.speculator.parameters():
            param.requires_grad = True

        self.speculator.to(device=self.device_name, dtype=self.torch_dtype)
        self.speculator.reset_parameters()

        if self.device_mesh.get_rank() == 0:
            print(f"Created MLP speculator with config: {speculator_config_dict}")

        return self.speculator

    def get_optimizer_params(self):
        speculator_module = self._get_speculator_module()
        if speculator_module is not None:
            return speculator_module.parameters()
        return None

    def _get_speculator_module(self):
        return self.speculator

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
        speculator_module = self._get_speculator_module()
        if speculator_module is None:
            return torch.tensor(0.0, device=self.device_name)

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        original_input_ids = input_ids
        input_ids = self._maybe_pad_nested(input_ids, padding=0)
        if loss_mask is not None:
            loss_mask = self._maybe_pad_nested(loss_mask, padding=0)
        if attention_mask is not None:
            attention_mask = self._maybe_pad_nested(attention_mask, padding=0)

        if hidden_states is None:
            with torch.no_grad():
                hidden_out = fsdp_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )
                hidden = hidden_out.hidden_states[-1]
        else:
            hidden = hidden_states
        hidden = self._maybe_unpack_packed_hidden(original_input_ids, attention_mask, hidden, packed_seq_params)
        hidden = self._maybe_normalize_hidden_layout(hidden, attention_mask, original_input_ids)
        hidden = self._maybe_pad_nested(hidden, padding=0.0)
        spec_dtype = next(speculator_module.parameters()).dtype
        if hidden.dtype != spec_dtype:
            hidden = hidden.to(dtype=spec_dtype)
        if spec_logits is None:
            spec_logits = self.compute_speculator_logits(input_ids, hidden)

        n_predict = speculator_module.n_predict
        vocab_size = spec_logits.size(-1)

        spec_loss_accum = 0.0
        loss_mask_matrix = loss_mask.reshape(input_ids.size(0), -1)
        for i in range(n_predict):
            start = i + 2
            length = spec_logits.size(2)
            max_len = min(
                length,
                input_ids.size(1) - start,
                loss_mask_matrix.size(1) - start,
            )
            if max_len <= 0:
                continue
            targets = input_ids[:, start : start + max_len]

            logits_i = spec_logits[i][:, :max_len, :].reshape(-1, vocab_size)
            labels_i = targets.reshape(-1)

            ce_i = loss_fct(logits_i, labels_i)
            mask_i = loss_mask_matrix[:, start : start + max_len].reshape(-1)
            ce_i = ce_i * mask_i
            spec_loss_accum += ce_i.sum() / mask_i.sum().clamp(min=1)

        spec_loss = spec_loss_accum / n_predict
        return spec_loss

    def compute_speculator_logits(self, input_ids, hidden_states):
        speculator_module = self._get_speculator_module()
        if speculator_module is None:
            return None

        # Speculator assumes regular tensors for slicing and indexing.
        if isinstance(input_ids, torch.Tensor) and input_ids.is_nested:
            input_ids = torch.nested.to_padded_tensor(input_ids, padding=0)
        if isinstance(hidden_states, torch.Tensor) and hidden_states.is_nested:
            hidden_states = torch.nested.to_padded_tensor(hidden_states, padding=0.0)

        n_predict = speculator_module.n_predict
        hidden, seq_ids = self._slice_speculator_inputs(input_ids, hidden_states, n_predict)
        pad_ids = torch.zeros(input_ids.size(0), n_predict, dtype=seq_ids.dtype, device=seq_ids.device)
        spec_inds = torch.cat([seq_ids, pad_ids], dim=1)

        spec_logits = speculator_module(hidden, spec_inds)
        return spec_logits

    # save_checkpoint/load_checkpoint use SpeculatorAdapter defaults
