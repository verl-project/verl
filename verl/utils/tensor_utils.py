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
"""Utility functions for tensor operations."""

from dataclasses import dataclass
from functools import reduce

import torch


@dataclass
class WeightChunkInfo:
    """Information about a chunk of a sliced weight tensor.

    Attributes:
        start_idx: Start index along the first dimension.
        end_idx: End index along the first dimension (exclusive).
        chunk_idx: Index of this chunk (0-based).
        total_chunks: Total number of chunks.
    """

    start_idx: int
    end_idx: int
    chunk_idx: int
    total_chunks: int


def compute_weight_chunks(
    name: str,
    weight: torch.Tensor,
    bucket_size: int,
) -> list[WeightChunkInfo]:
    """Compute how to slice a weight tensor into chunks that fit in bucket.

    This function calculates the chunking strategy for large weight tensors
    that exceed the bucket size. The tensor is sliced along its first dimension.

    Args:
        name: Name of the weight tensor (for error messages and logging).
        weight: The weight tensor to slice.
        bucket_size: Maximum size in bytes for each chunk.

    Returns:
        List of WeightChunkInfo, one for each chunk.
        Returns a single-element list if the weight doesn't need slicing.

    Raises:
        ValueError: If a single slice is larger than bucket_size.
    """
    import logging

    logger = logging.getLogger(__name__)

    weight_size = weight.nbytes
    if weight_size <= bucket_size:
        # No slicing needed
        return [WeightChunkInfo(start_idx=0, end_idx=weight.shape[0], chunk_idx=0, total_chunks=1)]

    # Slice the weight along the first dimension into chunks
    dtype_size = weight.element_size()
    numel_per_chunk = bucket_size // dtype_size

    # Calculate chunk size along the first dimension
    first_dim_size = weight.shape[0]
    elements_per_row = reduce(lambda x, y: x * y, weight.shape[1:], 1)
    if elements_per_row == 0:
        # Empty tensor, return as is
        return [WeightChunkInfo(start_idx=0, end_idx=first_dim_size, chunk_idx=0, total_chunks=1)]

    chunk_dim_size = numel_per_chunk // elements_per_row
    if chunk_dim_size == 0:
        raise ValueError(
            f"Weight '{name}' with shape {weight.shape} is too large to be chunked. A single slice "
            f"along the first dimension is larger than the bucket size ({bucket_size} bytes). "
            f"Please increase `checkpoint_engine.update_weights_bucket_megabytes`."
        )

    num_chunks = (first_dim_size + chunk_dim_size - 1) // chunk_dim_size
    logger.info(f"Slicing weight {name} ({weight.shape}, {weight.dtype}, {weight_size} bytes) into {num_chunks} chunks")

    chunks = []
    start_idx = 0
    for chunk_idx in range(num_chunks):
        end_idx = min(start_idx + chunk_dim_size, first_dim_size)
        chunks.append(
            WeightChunkInfo(
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_idx=chunk_idx,
                total_chunks=num_chunks,
            )
        )
        start_idx = end_idx

    return chunks
