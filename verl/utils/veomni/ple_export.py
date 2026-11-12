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

"""Restore checkpoint PLE shard shapes after gathering training partitions."""


def ple_export_shapes(model):
    """Use runtime checkpoint boundaries, excluding per-shard training padding."""
    shapes = {}
    for prefix, module in model.named_modules():
        if not all(
            hasattr(module, attr)
            for attr in ("rows_per_checkpoint_shard", "padded_rows_per_shard", "split_ngram_parts", "ngram_embedding")
        ):
            continue
        for index in range(module.split_ngram_parts):
            embedding = module.ngram_embedding[f"shard_{index}"]
            name = f"{prefix}.ngram_embedding.shard_{index}.weight".lstrip(".")
            shapes[name] = (
                module.rows_per_checkpoint_shard,
                module.padded_rows_per_shard,
                embedding.embedding_dim,
            )
    return shapes


def trim_ple_export(name, tensor, shapes):
    """Trim only known trailing padding; reject unexpected training layouts."""
    if name not in shapes:
        raise ValueError(f"Missing PLE export metadata for {name}")
    rows, padded_rows, columns = shapes[name]
    if rows <= 0 or padded_rows < rows or tuple(tensor.shape) != (padded_rows, columns):
        raise ValueError(
            f"Invalid PLE export shape for {name}: expected {(padded_rows, columns)}, "
            f"got {tuple(tensor.shape)}, checkpoint rows={rows}"
        )
    return tensor[:rows]
