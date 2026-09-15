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
"""DeepSeek-V4 tensors that SGLang fuses within one ``load_weights`` call."""

_FUSION_MEMBERS = (
    ("wq_a.weight", "wkv.weight"),
    ("wq_a.scale", "wkv.scale"),
    ("wq_a.weight_scale_inv", "wkv.weight_scale_inv"),
    ("compressor.wkv.weight", "compressor.wgate.weight"),
    ("indexer.compressor.wkv.weight", "indexer.compressor.wgate.weight"),
)

DEEPSEEK_V4_FUSION_GROUPS = tuple(
    tuple(f".{attention}.{member}" for member in members)
    for attention in ("self_attn", "attn")
    for members in _FUSION_MEMBERS
)
