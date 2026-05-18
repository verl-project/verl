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

"""Runtime patches mirroring local MindSpeed modifications."""

import logging

logger = logging.getLogger(__name__)


def apply_mindspeed_train_infer_consist_patches() -> None:
    logger.info("Applying MindSpeed train-inference consistency patches...")
    _patch_flash_attention_adaptor()
    _patch_layernorm_column_parallel_linear()


def _patch_flash_attention_adaptor() -> None:
    """Patch adaptor.dot_product_attention_forward_impl with FA3 branch."""
    try:
        import os
        import torch
        from flash_attn import flash_attn_varlen_func
        import mindspeed.core.transformer.flash_attention.flash_attention.adaptor as adaptor

        if hasattr(adaptor, "_orig_dot_product_attention_forward_impl_train_infer_consist"):
            return

        original_impl = adaptor.dot_product_attention_forward_impl

        def _patched_impl(self, query, key, value, attention_mask,
                          attn_mask_type=None, attention_bias=None,
                          packed_seq_params=None):
            if packed_seq_params is not None and os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1":
                scale = self.softmax_scale
                cu_seqlens_q = torch.tensor(
                    packed_seq_params.cu_seqlens_q, dtype=torch.int32).npu()
                cu_seqlens_k = torch.tensor(
                    packed_seq_params.cu_seqlens_kv, dtype=torch.int32).npu()
                max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
                max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max())

                q_fa = query.detach().clone()
                k_fa = key.detach().clone()
                v_fa = value.detach().clone()
                q_fa.requires_grad = True
                k_fa.requires_grad = True
                v_fa.requires_grad = True
                return flash_attn_varlen_func(
                    q_fa,
                    k_fa,
                    v_fa,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    dropout_p=self.attention_dropout.p,
                    softmax_scale=scale,
                    causal=True,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=False,
                    block_table=None,
                )
            return original_impl(
                self,
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        adaptor._orig_dot_product_attention_forward_impl_train_infer_consist = original_impl
        adaptor.dot_product_attention_forward_impl = _patched_impl
    except Exception as e:
        logger.warning("Failed to patch MindSpeed adaptor: %s", e)


def _patch_layernorm_column_parallel_linear() -> None:
    """Patch te_return_bias and _rmsnorm behavior."""
    try:
        import torch_npu
        from mindspeed.te.pytorch.module.layernorm_column_parallel_linear import (
            MindSpeedTELayerNormColumnParallelLinear,
        )

        if hasattr(MindSpeedTELayerNormColumnParallelLinear, "_patched_train_infer_consist"):
            return

        original_init = MindSpeedTELayerNormColumnParallelLinear.__init__
        original_rmsnorm = MindSpeedTELayerNormColumnParallelLinear._rmsnorm

        def _patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Align with local modification:
            # self.te_return_bias = (not self.skip_bias_add) and bias
            self.te_return_bias = (not self.skip_bias_add) and (self.bias is not None)

        def _patched_rmsnorm(self, inp):
            return torch_npu.npu_rms_norm(
                inp, self.layer_norm_weight, epsilon=self.config.layernorm_epsilon
            )[0]

        MindSpeedTELayerNormColumnParallelLinear.__init__ = _patched_init
        MindSpeedTELayerNormColumnParallelLinear._rmsnorm = _patched_rmsnorm
        MindSpeedTELayerNormColumnParallelLinear._orig_init_train_infer_consist = original_init
        MindSpeedTELayerNormColumnParallelLinear._orig_rmsnorm_train_infer_consist = original_rmsnorm
        MindSpeedTELayerNormColumnParallelLinear._patched_train_infer_consist = True
    except Exception as e:
        logger.warning("Failed to patch layernorm_column_parallel_linear: %s", e)
