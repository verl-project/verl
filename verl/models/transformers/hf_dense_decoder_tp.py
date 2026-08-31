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

"""Conventional HF tensor-parallel placement for dense decoder weights."""

from __future__ import annotations


def infer_dense_decoder_tp_shard_dim(name: str) -> int | None:
    """Infer the conventional HF tensor-parallel placement from a weight name."""

    if name.endswith(
        (
            "embed_tokens.weight",
            "lm_head.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
        )
    ):
        return 0
    if name.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight")):
        return 1
    if name.endswith("norm.weight"):
        return None
    raise NotImplementedError(f"unvalidated dense-decoder TP parameter: {name}")
