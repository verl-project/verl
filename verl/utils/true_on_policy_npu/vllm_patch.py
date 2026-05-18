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
Train-Inference Consistency Patches for vLLM

Patches applied:
1. RowParallelLinear.forward() - Change all-reduce to reduce-scatter + all-gather

Source code location: vllm/model_executor/layers/linear.py lines 1407-1426
"""

import logging

logger = logging.getLogger(__name__)


def apply_vllm_train_infer_consist_patches() -> None:
    """Apply all train-inference consistency patches for vLLM."""
    logger.info("Applying vLLM train-inference consistency patches...")

    _patch_row_parallel_linear()

    logger.info("vLLM train-inference consistency patches applied.")


def _patch_row_parallel_linear() -> None:
    """
    Patch RowParallelLinear to use reduce-scatter + all-gather instead of all-reduce.

    This matches MindSpeed's layernorm_column_parallel_linear.py training pattern.
    """
    try:
        from vllm.model_executor.layers.linear import RowParallelLinear
        from vllm.distributed import (
            tensor_model_parallel_reduce_scatter,
            tensor_model_parallel_all_gather,
        )
        import torch.nn.functional as F

        if hasattr(RowParallelLinear, '_patched_for_train_infer_consist'):
            logger.debug("RowParallelLinear already patched.")
            return

        RowParallelLinear._original_forward = RowParallelLinear.forward

        def _patched_forward(self, input_):
            """Patched forward using RS+AG pattern."""
            # Input handling
            if self.input_is_parallel:
                input_parallel = input_
            else:
                from vllm.model_executor.layers.linear import split_tensor_along_last_dim
                splitted_input = split_tensor_along_last_dim(
                    input_, num_partitions=self.tp_size
                )
                input_parallel = splitted_input[self.tp_rank].contiguous()

            # Matrix multiply
            bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
            output_parallel = self.quant_method.apply(self, input_parallel, bias_)

            # MODIFIED: Use reduce-scatter + all-gather instead of all-reduce
            if self.reduce_results and self.tp_size > 1:
                # Calculate padding for divisibility
                pad_size = (self.tp_size - (output_parallel.shape[0] % self.tp_size)) % self.tp_size

                if pad_size > 0:
                    output_parallel = F.pad(output_parallel, (0, 0, 0, pad_size))

                scattered = tensor_model_parallel_reduce_scatter(output_parallel, dim=0)
                output = tensor_model_parallel_all_gather(scattered, dim=0)

                if pad_size > 0:
                    output = output[:-pad_size]
            else:
                output = output_parallel

            output_bias = self.bias if self.skip_bias_add else None

            if not self.return_bias:
                return output
            return output, output_bias

        RowParallelLinear.forward = _patched_forward
        RowParallelLinear._patched_for_train_infer_consist = True

        logger.info("Patched RowParallelLinear to use reduce-scatter + all-gather.")

    except Exception as e:
        logger.warning(f"Failed to patch RowParallelLinear: {e}")
