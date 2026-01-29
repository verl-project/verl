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

import types

import torch
from torch import nn
from verl.trainer.speculators.interface import SpeculatorAdapter
import transformers
from verl.trainer.speculators.config import SpeculatorConfigBase
import os

class ArcticLSTMSpeculatorConfig(SpeculatorConfigBase):
    """
    Configs used for saving model checkpoint for inference.
    """

    def __init__(
        self,
        base_model_name_or_path,
        input_hidden_dim,
        inner_dim,
        proj_dim,
        emb_dim,
        vocab_size,
        n_predict,
        tie_weights=False,
        scale_input=False,
        method="sum_rnn",
        tie_lstm_embs=False,
    ):
        self.architectures = ["ArcticLSTMSpeculatorPreTrainedModel"]
        self.base_model_name_or_path = base_model_name_or_path

        self.input_hidden_dim = input_hidden_dim
        self.inner_dim = str(inner_dim)
        self.proj_dim = str(proj_dim)
        self.emb_dim = str(emb_dim)
        self.model_type = "mlp_speculator"

        self.n_candidates = n_predict
        self.n_predict = n_predict

        self.scale_input = scale_input
        self.tie_weights = tie_weights
        self.tie_lstm_embs = tie_lstm_embs
        self.top_k_tokens_per_head = [1 for _ in range(self.n_predict)]

        self.torch_dtype = "bfloat16"
        self.transformers_version = transformers.__version__
        self.vocab_size = vocab_size
        self.method = method



class LSTMSpeculatorAdapter(SpeculatorAdapter):
    def __init__(
        self,
        config,
        model_config,
        device_name,
        device_mesh,
        torch_dtype,
        speculator_config=None,
    ):
        self.config = config
        self.model_config = model_config
        self.device_name = device_name
        self.device_mesh = device_mesh
        self.torch_dtype = torch_dtype

        if speculator_config is None:
            if self.config is not None and hasattr(self.config, "model"):
                speculator_config = getattr(self.config.model, "speculator", None)
            else:
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
            "input_hidden_dim": hidden_size,
            "inner_dim": str(self.speculator_config.get("inner_dim", hidden_size)),
            "emb_dim": str(self.speculator_config.get("emb_dim", hidden_size)),
            "proj_dim": str(self.speculator_config.get("proj_dim", hidden_size)),
            "vocab_size": vocab_size,
            "scale_input": self.speculator_config.get("scale_input", False),
            "tie_weights": self.speculator_config.get("tie_weights", False),
            "tie_lstm_embs": self.speculator_config.get("tie_lstm_embs", False),
            "method": self.speculator_config.get("method", "sum_lstm"),
        }

        from verl.models.speculator import speculator as speculator_mod

        if hasattr(speculator_mod, "create_speculator_from_config"):
            self.speculator = speculator_mod.create_speculator_from_config(speculator_config_dict)
        else:
            config_obj = types.SimpleNamespace(**speculator_config_dict)
            self.speculator = speculator_mod.ArcticLSTMSpeculator(config_obj)

        for param in model.parameters():
            param.requires_grad = False
        for param in self.speculator.parameters():
            param.requires_grad = True

        self.speculator.to(device=self.device_name, dtype=self.torch_dtype)
        self.speculator.reset_parameters()
        if self.device_mesh.get_rank() == 0:
            print(f"Created speculator with config: {speculator_config_dict}")

        return self.speculator



    def _get_speculator_config_obj(self, speculator_module):
        if speculator_module is None:
            return None
        base_model_name_or_path = None
        if self.config is not None and hasattr(self.config, "model"):
            base_model_name_or_path = getattr(self.config.model, "partial_pretrain", None)
        if base_model_name_or_path is None and self.model_config is not None:
            base_model_name_or_path = getattr(self.model_config, "local_path", None)
        return ArcticLSTMSpeculatorConfig(
            base_model_name_or_path=base_model_name_or_path,
            input_hidden_dim=speculator_module.input_hidden_dim,
            inner_dim=speculator_module.inner_dim,
            proj_dim=speculator_module.proj_dim,
            emb_dim=speculator_module.emb_dim,
            vocab_size=speculator_module.vocab_size,
            n_predict=speculator_module.n_predict,
            tie_weights=speculator_module.tie_weights,
            scale_input=speculator_module.scale_input,
            method=speculator_module.method,
            tie_lstm_embs=speculator_module.tie_lstm_embs,
        )

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

        n_predict = speculator_module.n_predict
        hidden, seq_ids = self._slice_speculator_inputs(input_ids, hidden_states, n_predict)
        pad_ids = torch.zeros(input_ids.size(0), n_predict, dtype=seq_ids.dtype, device=seq_ids.device)
        spec_inds = torch.cat([seq_ids, pad_ids], dim=1)

        spec_logits = speculator_module(hidden, spec_inds)
        return spec_logits

    # save_checkpoint/load_checkpoint use SpeculatorAdapter defaults
