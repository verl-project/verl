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
"""Kimi output-token support shared with the rollout logits mask."""

import torch


def is_kimi_model(hf_config):
    return getattr(hf_config, "model_type", "") in {"kimi_k3", "kimi_k31"}


def mask_kimi_generation_logits(logits: torch.Tensor, hf_config):
    if not is_kimi_model(hf_config):
        return logits
    token_id = getattr(hf_config, "media_placeholder_token_id", None)
    if token_id is None or not 0 <= token_id < logits.shape[-1]:
        raise ValueError("Kimi generation mask requires a valid media_placeholder_token_id")
    if hasattr(logits, "full_tensor"):
        logits = logits.full_tensor()
    # Match vLLM's get_vision_placeholder_token_ids/monkey_patch_compute_logits.
    # The placeholder has image features on input, but cannot be generated.
    mask = torch.arange(logits.shape[-1], device=logits.device) == token_id
    return logits.masked_fill(mask, -torch.inf)
