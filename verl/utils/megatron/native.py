# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# there is some bug in mcore 0.12, so we need to patch it
# 1. `get_query_key_value_tensors` in `multi_latent_attention.py` works wrong when packed_seq_params is not None


"""Resolve the language component for native output callbacks."""


def get_language_model(model):
    """Unwrap distributed containers and then a multimodal language component."""
    while hasattr(model, "module"):
        model = model.module
    return getattr(model, "language_model", model)
