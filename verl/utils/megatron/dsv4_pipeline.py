# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Pipeline stage sizing for DeepSeek-V4 hybrid models (no Megatron imports, so it is unit-testable on CPU)."""


def dsv4_logical_layers_per_stage(num_logical_layers: int, num_stages: int, num_hash_layers: int) -> list[int] | None:
    """Logical (attention + MoE) blocks per pipeline stage, in VPP-major / PP-minor order.

    Megatron-Core requires every hash-routed MoE layer to live on the stage that owns the embedding, because
    only that stage has ``input_ids``. An even split satisfies this whenever the first stage already holds
    ``num_hash_layers`` blocks; then ``None`` is returned and the caller keeps the default even split. Otherwise
    the first stage is grown to ``num_hash_layers`` blocks, taking one block at a time from the fullest later
    stage (the last such stage on ties).
    """
    if num_stages <= 1 or num_logical_layers <= 0:
        return None
    base, extra = divmod(num_logical_layers, num_stages)
    counts = [base + int(stage < extra) for stage in range(num_stages)]
    need = min(max(num_hash_layers, 0), num_logical_layers)
    if counts[0] >= need:
        return None
    if num_logical_layers - need < num_stages - 1:
        raise ValueError(
            f"Cannot place {need} hash MoE layers on the first of {num_stages} pipeline stages and still give every "
            f"other stage a layer with only {num_logical_layers} DeepSeek-V4 layers; lower PP/VPP or add layers."
        )
    while counts[0] < need:
        donor = max(range(num_stages - 1, 0, -1), key=lambda stage: counts[stage])
        counts[donor] -= 1
        counts[0] += 1
    return counts
