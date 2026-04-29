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
Train-Inference Consistency Patches for vLLM-Ascend (NPU)

This module applies runtime patches to ensure vLLM-Ascend inference behavior
matches Megatron/MindSpeed training behavior.

Source code fixes required in vllm-ascend:
1. vllm_ascend/ops/fused_moe/moe_mlp.py - Import order fix (torch.nn.functional as F before torch_npu)
2. vllm_ascend/ops/activation.py - SwiGLU implementation (SiLU-based, lines 43-44)
3. vllm_ascend/ascend_forward_context.py - MoE ALLTOALL override (lines 284-285)
4. vllm_ascend/attention/attention_v1.py - FlashAttention with batch invariant (lines 590-660)
5. vllm_ascend/ops/fused_moe/token_dispatcher.py - Import order and AllToAll communication patterns
"""

import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def apply_batch_invariance_runtime_patches() -> None:
    """
    Apply vllm-ascend batch-invariant runtime patches.

    This mirrors the local vllm_ascend/batch_invariant.py behavior and can be
    called from both training and inference side.
    """
    os.environ["VLLM_BATCH_INVARIANT"] = "1"

    _patch_batch_invariant_module()

    try:
        from vllm_ascend.batch_invariant import init_batch_invariance
        init_batch_invariance()
    except Exception as e:
        logger.warning("init_batch_invariance failed: %s", e)


def apply_vllm_ascend_train_infer_consist_patches() -> None:
    """Apply runtime patches that mirror local vllm-ascend changes."""
    apply_batch_invariance_runtime_patches()

    _patch_ascend_forward_context()
    _patch_swiglu_activation()
    _patch_moe_mlp_unquant_apply_mlp()
    _patch_attention_impl()
    _patch_token_dispatcher_weights()


def _patch_batch_invariant_module() -> None:
    """Patch vllm_ascend.batch_invariant to include newly added operators."""
    try:
        import torch_npu
        import vllm_ascend.batch_invariant as bi

        if hasattr(bi, "_patched_batch_invariant_train_infer_consist"):
            return

        original_torch_sum = torch.sum
        original_tensor_sum = torch.Tensor.sum

        has_ascendc_batch_invariant = False
        try:
            import batch_invariant_ops  # type: ignore[import-not-found] # noqa: F401
            has_ascendc_batch_invariant = True
        except Exception:
            has_ascendc_batch_invariant = False

        def add_rms_norm(x: torch.Tensor,
                         residual: torch.Tensor,
                         weight: torch.Tensor,
                         eps: float):
            x_ = x + residual
            residual_ = x_
            x_, _ = torch_npu.npu_rms_norm(x_, weight, eps)
            return x_, None, residual_

        def reduce_sum_adapter(x: torch.Tensor, dim=None, keepdim=False,
                               dtype=None, **kwargs):
            if "axis" in kwargs:
                dim = kwargs.pop("axis")

            should_use_batch_invariant = (
                x.device.type == "npu"
                and x.dtype.is_floating_point
                and dim is not None
                and has_ascendc_batch_invariant
            )
            if should_use_batch_invariant:
                if isinstance(dim, (tuple, list)):
                    if len(dim) == 1:
                        dim = dim[0]
                    else:
                        return original_tensor_sum(
                            x, dim=dim, keepdim=keepdim, dtype=dtype, **kwargs
                        )
                return torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(
                    x, dim, keepdim
                )

            return original_tensor_sum(x, dim=dim, keepdim=keepdim, dtype=dtype, **kwargs)

        def softmax_adapter(x, dim, dtype=None):
            return torch.ops.batch_invariant_ops.npu_softmax_batch_invariant(x, dim)

        def log_softmax_adapter(x, dim, dtype=None):
            return torch.ops.batch_invariant_ops.npu_log_softmax_batch_invariant(x, dim)

        def override_envs_for_invariance():
            os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"
            os.environ["HCCL_DETERMINISTIC"] = "strict"
            os.environ["LCCL_DETERMINISTIC"] = "1"

        def enable_batch_invariant_mode():
            bi._batch_invariant_LIB = torch.library.Library("aten", "IMPL")

            if getattr(bi, "HAS_TRITON", False):
                bi._batch_invariant_LIB.impl("aten::addmm", bi.addmm_batch_invariant, "NPU")
                bi._batch_invariant_LIB.impl("aten::bmm", bi.bmm_batch_invariant, "NPU")

            if has_ascendc_batch_invariant:
                bi._batch_invariant_LIB.impl(
                    "aten::mm",
                    torch.ops.batch_invariant_ops.npu_mm_batch_invariant,
                    "NPU",
                )
                bi._batch_invariant_LIB.impl(
                    "aten::matmul",
                    torch.ops.batch_invariant_ops.npu_matmul_batch_invariant,
                    "NPU",
                )
                bi._batch_invariant_LIB.impl("aten::softmax", softmax_adapter, "NPU")
                bi._batch_invariant_LIB.impl("aten::_softmax", softmax_adapter, "NPU")
                bi._batch_invariant_LIB.impl("aten::_log_softmax", log_softmax_adapter, "NPU")
                bi._batch_invariant_LIB.impl("aten::log_softmax", log_softmax_adapter, "NPU")
                torch_npu.npu_fused_infer_attention_score = (
                    torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant
                )
                torch_npu.npu_add_rms_norm = add_rms_norm
                torch.sum = reduce_sum_adapter
                torch.Tensor.sum = reduce_sum_adapter
            elif getattr(bi, "HAS_TRITON", False):
                bi._batch_invariant_LIB.impl("aten::mm", bi.mm_batch_invariant, "NPU")
                bi._batch_invariant_LIB.impl("aten::matmul", bi.matmul_batch_invariant, "NPU")
                bi._batch_invariant_LIB.impl("aten::linear", bi.linear_batch_invariant, "NPU")

        def init_batch_invariance():
            if bi.vllm_is_batch_invariant():
                if getattr(bi, "HAS_TRITON", False) or has_ascendc_batch_invariant:
                    logger.info("Enabling batch-invariant mode for vLLM on Ascend NPU.")
                    override_envs_for_invariance()
                    enable_batch_invariant_mode()
                else:
                    logger.warning(
                        "Batch-invariant mode requested but Triton/Ascend batch-invariant "
                        "ops are unavailable, skipping initialization."
                    )

        bi._orig_torch_sum_train_infer_consist = original_torch_sum
        bi._orig_tensor_sum_train_infer_consist = original_tensor_sum
        bi.HAS_ASCENDC_BATCH_INVARIANT = has_ascendc_batch_invariant
        bi.add_rms_norm = add_rms_norm
        bi.reduce_sum_adapter = reduce_sum_adapter
        bi.softmax_adapter = softmax_adapter
        bi.log_softmax_adapter = log_softmax_adapter
        bi.override_envs_for_invariance = override_envs_for_invariance
        bi.enable_batch_invariant_mode = enable_batch_invariant_mode
        bi.init_batch_invariance = init_batch_invariance
        bi._patched_batch_invariant_train_infer_consist = True
    except Exception as e:
        logger.warning("Failed to patch batch_invariant module: %s", e)


def _patch_swiglu_activation() -> None:
    """
    Patch AscendSiluAndMul to use SiLU-based SwiGLU matching training behavior.

    Source code location: vllm_ascend/ops/activation.py lines 43-44
    """
    try:
        from vllm_ascend.ops.activation import AscendSiluAndMul

        if hasattr(AscendSiluAndMul, '_patched_for_train_infer_consist'):
            return

        original_forward_oot = AscendSiluAndMul.forward_oot

        def _patched_forward_oot(self, x):
            d = x.shape[-1] // 2
            return F.silu(x[..., :d]) * x[..., d:]

        AscendSiluAndMul.forward_oot = _patched_forward_oot
        AscendSiluAndMul._patched_for_train_infer_consist = True
        AscendSiluAndMul._original_forward_oot = original_forward_oot

    except Exception as e:
        logger.warning("Failed to patch AscendSiluAndMul: %s", e)


def _patch_moe_mlp_unquant_apply_mlp() -> None:
    """Patch moe_mlp.unquant_apply_mlp non-310P path to SiLU split form."""
    try:
        import vllm_ascend.ops.fused_moe.moe_mlp as moe_mlp
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if hasattr(moe_mlp, "_orig_unquant_apply_mlp_train_infer_consist"):
            return

        original = moe_mlp.unquant_apply_mlp

        def _patched_unquant_apply_mlp(hidden_states, w1, w2, group_list,
                                       group_list_type=1, topk_scales=None,
                                       need_trans=True):
            if need_trans:
                w1_local = w1.transpose(1, 2)
                w2_local = w2.transpose(1, 2)
            else:
                w1_local = w1
                w2_local = w2

            import torch_npu
            gate_up_out = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[w1_local],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
            )[0]
            if get_ascend_device_type() == AscendDeviceType._310P:
                gate_up_out = torch_npu.npu_swiglu(
                    gate_up_out.to(torch.float32)).to(torch.float16)
            else:
                d = gate_up_out.shape[-1] // 2
                gate_up_out = F.silu(gate_up_out[..., :d]) * gate_up_out[..., d:]

            if topk_scales is not None:
                gate_up_out *= topk_scales

            out = torch_npu.npu_grouped_matmul(
                x=[gate_up_out],
                weight=[w2_local],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
            )[0]
            return out

        moe_mlp._orig_unquant_apply_mlp_train_infer_consist = original
        moe_mlp.unquant_apply_mlp = _patched_unquant_apply_mlp
    except Exception as e:
        logger.warning("Failed to patch moe_mlp.unquant_apply_mlp: %s", e)


def _patch_ascend_forward_context() -> None:
    """Force ALLTOALL when TRAIN_INFER_CONSIST=1."""
    try:
        import vllm_ascend.ascend_forward_context as afc

        if hasattr(afc, "_orig_select_moe_comm_method_train_infer_consist"):
            return

        original = afc.select_moe_comm_method

        def _patched_select_moe_comm_method(num_tokens, vllm_config,
                                            is_draft_model=False):
            moe_comm_type = original(num_tokens, vllm_config, is_draft_model)
            if os.environ.get("TRAIN_INFER_CONSIST", "0") == "1":
                return afc.MoECommType.ALLTOALL
            return moe_comm_type

        afc._orig_select_moe_comm_method_train_infer_consist = original
        afc.select_moe_comm_method = _patched_select_moe_comm_method
    except Exception as e:
        logger.warning("Failed to patch ascend_forward_context: %s", e)


def _patch_attention_impl() -> None:
    """Patch forward_fused_infer_attention with FA3 path for batch-invariant mode."""
    try:
        import vllm_ascend.attention.attention_v1 as attention_v1
        AscendAttentionBackendImpl = attention_v1.AscendAttentionBackendImpl

        if hasattr(AscendAttentionBackendImpl,
                   "_patched_forward_fused_train_infer_consist"):
            return

        original_forward = AscendAttentionBackendImpl.forward_fused_infer_attention

        def _patched_forward_fused_infer_attention(
            self, query, key, value, attn_metadata, output
        ):
            import torch_npu

            forward_context = attention_v1.get_forward_context()
            if getattr(forward_context, "capturing", False):
                attn_output, num_tokens = self.full_graph_fia(
                    query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output

            if (
                attn_metadata.attn_state == attention_v1.AscendAttentionState.DecodeOnly
                and self.sliding_window is not None
                and attn_metadata.seq_lens.shape[0] == query.size(0)
            ):
                return self._forward_fia_slidingwindow(query, attn_metadata, output)

            if (
                attn_metadata.attn_state == attention_v1.AscendAttentionState.PrefillNoCache
                and self.attn_type != attention_v1.AttentionType.ENCODER_DECODER
                and os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
            ):
                num_tokens = attn_metadata.actual_seq_lengths_q[-1]
                query = query[:num_tokens]
                key = key[:num_tokens]
                value = value[:num_tokens]

                num_decodes = attn_metadata.num_decodes
                num_decode_tokens = attn_metadata.num_decode_tokens
                num_prefills = attn_metadata.num_prefills
                attn_output_fa_decode = None
                attn_output_fa_prefill = None

                if num_decodes > 0:
                    seq_lens_list_qa = attn_metadata.seq_lens_list[:num_decodes]
                    actual_seq_lengths_fa = attn_metadata.query_start_loc[:num_decodes + 1]
                    max_seq_len = max(seq_lens_list_qa)
                    attn_output_fa_decode = self._foward_fa3(
                        query[:num_decode_tokens],
                        attn_metadata.block_tables[:num_decodes, :],
                        actual_seq_lengths_fa,
                        seq_lens_list_qa,
                        False,
                        max_seq_len,
                    )

                if num_prefills > 0:
                    seq_lens_list_qa = attn_metadata.seq_lens_list[num_decodes:]
                    actual_seq_lengths_fa = attn_metadata.query_start_loc[num_decodes:]
                    max_seq_len = max(seq_lens_list_qa)
                    attn_output_fa_prefill = self._foward_fa3(
                        query[num_decode_tokens:],
                        attn_metadata.block_tables[num_decode_tokens:, :],
                        actual_seq_lengths_fa,
                        seq_lens_list_qa,
                        True,
                        max_seq_len,
                    )

                outputs = []
                if attn_output_fa_decode is not None:
                    outputs.append(attn_output_fa_decode)
                if attn_output_fa_prefill is not None:
                    outputs.append(attn_output_fa_prefill)
                if len(outputs) == 0:
                    raise ValueError("No attention output available")
                attn_output_fa = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
                output[:num_tokens] = attn_output_fa[:num_tokens]
                return output

            key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
                key, value, attn_metadata)
            num_tokens = attn_metadata.actual_seq_lengths_q[-1]
            query = query[:num_tokens]
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
            output[:num_tokens] = attn_output[:num_tokens]
            return output

        def _patched_foward_fa3(
            self, query, block_table, actual_seq_lengths_fa, seq_lens_list_qa, is_causal, max_seq_len
        ):
            from flash_attn_v3 import flash_attn_with_kvcache

            window_size_left = -1
            window_size_right = -1
            num_splits = 0
            rotary_cos = None
            rotary_sin = None
            cache_batch_idx = None
            leftpad_k = None
            num_block, block_size, _, _ = self.key_cache.shape
            key_fa_blk = self.key_cache.view(num_block, block_size, self.num_kv_heads, self.head_size)
            value_fa_blk = self.value_cache.view(num_block, block_size, self.num_kv_heads, self.head_size)
            kv_seqlen_list = torch.tensor(seq_lens_list_qa, dtype=torch.int32).npu()
            return flash_attn_with_kvcache(
                query,
                key_fa_blk,
                value_fa_blk,
                None,
                None,
                None,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                cache_seqlens=kv_seqlen_list,
                cache_batch_idx=cache_batch_idx,
                cache_leftpad=leftpad_k,
                page_table=block_table,
                cu_seqlens_q=actual_seq_lengths_fa,
                cu_seqlens_k_new=None,
                max_seqlen_q=max_seq_len,
                rotary_seqlens=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                softmax_scale=None,
                causal=is_causal,
                window_size=[window_size_left, window_size_right],
                attention_chunk=0,
                softcap=0.0,
                rotary_interleaved=False,
                scheduler_metadata=None,
                num_splits=num_splits,
                pack_gqa=None,
                sm_margin=0,
                return_softmax_lse=False,
            )

        AscendAttentionBackendImpl._orig_forward_fused_infer_attention_train_infer_consist = (
            original_forward
        )
        AscendAttentionBackendImpl.forward_fused_infer_attention = (
            _patched_forward_fused_infer_attention
        )
        AscendAttentionBackendImpl._orig_foward_fa3_train_infer_consist = getattr(
            AscendAttentionBackendImpl, "_foward_fa3", None
        )
        AscendAttentionBackendImpl._foward_fa3 = _patched_foward_fa3
        AscendAttentionBackendImpl._patched_forward_fused_train_infer_consist = True
    except Exception as e:
        logger.warning("Failed to patch attention impl: %s", e)


def _patch_token_dispatcher_weights() -> None:
    """
    Patch TokenDispatcherWithAll2AllV to handle topk_weights via AllToAll.

    This ensures topk weights are properly permuted and communicated alongside
    hidden states to match training behavior.

    """
    try:
        from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV

        if hasattr(TokenDispatcherWithAll2AllV, '_patched_for_weights'):
            return

        original_token_dispatch = TokenDispatcherWithAll2AllV.token_dispatch

        def _patched_token_dispatch(self, hidden_states, topk_weights, topk_ids,
                                    expert_map=None, global_redundant_expert_num=0,
                                    mc2_mask=None, apply_router_weight_on_input=False,
                                    with_quant=False, dynamic_eplb=False,
                                    pertoken_scale=None):
            """Patched dispatch with topk_weights handling via AllToAll."""
            from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all
            import torch_npu

            self.with_quant = with_quant
            self.hidden_shape = hidden_states.shape

            permutated_local_input_tokens, reversed_local_input_permutation_mapping, \
                tokens_per_expert, input_splits, output_splits, \
                num_global_tokens_per_local_expert, global_input_tokens_local_experts_indices = \
                self._dispatch_preprocess(hidden_states, topk_ids)

            num_permuted = permutated_local_input_tokens.shape[0]

            dynamic_scale_after_all2all = None
            if self.with_quant:
                permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                    permutated_local_input_tokens)
                _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
                    dynamic_scale, output_splits, input_splits, self.ep_group)
                permute2_ep_all_to_all_handle.wait()
                dynamic_scale.untyped_storage().resize_(0)

            _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
                permutated_local_input_tokens, output_splits, input_splits, self.ep_group)
            permute1_ep_all_to_all_handle.wait()
            permutated_local_input_tokens.untyped_storage().resize_(0)

            flat_weights = topk_weights.view(-1)
            permuted_weights = torch.zeros(
                num_permuted, 1,
                dtype=flat_weights.dtype,
                device=flat_weights.device)
            permuted_weights.scatter_(
                0,
                reversed_local_input_permutation_mapping.unsqueeze(-1).long(),
                flat_weights.unsqueeze(-1))

            global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = \
                self._dispatch_postprocess(
                    global_input_tokens, dynamic_scale_after_all2all,
                    global_input_tokens_local_experts_indices)

            _, global_weights, weights_handle = async_all_to_all(
                permuted_weights, output_splits, input_splits, self.ep_group)
            weights_handle.wait()
            permuted_weights.untyped_storage().resize_(0)

            if self.num_local_experts > 1 and global_input_tokens_local_experts_indices is not None:
                global_weights, _ = torch_npu.npu_moe_token_permute(
                    global_weights, global_input_tokens_local_experts_indices)

            context_metadata = {
                "input_splits": input_splits,
                "output_splits": output_splits,
                "topk_weights": topk_weights,
                "reversed_local_input_permutation_mapping": reversed_local_input_permutation_mapping,
                "reversed_global_input_permutation_mapping": reversed_global_input_permutation_mapping
            }

            from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatchResult
            return TokenDispatchResult(
                hidden_states=global_input_tokens,
                dynamic_scale=dynamic_scale_final,
                group_list=tokens_per_expert,
                group_list_type=1,
                context_metadata=context_metadata,
                topk_scales=global_weights,
            )

        TokenDispatcherWithAll2AllV.token_dispatch = _patched_token_dispatch
        TokenDispatcherWithAll2AllV._patched_for_weights = True
        TokenDispatcherWithAll2AllV._original_token_dispatch = original_token_dispatch

        original_combine_postprocess = TokenDispatcherWithAll2AllV._combine_postprocess

        def _patched_combine_postprocess(self, permutated_local_input_tokens, context_metadata):
            import torch_npu

            output = torch_npu.npu_moe_token_unpermute(
                permuted_tokens=permutated_local_input_tokens,
                sorted_indices=context_metadata[
                    "reversed_local_input_permutation_mapping"].to(torch.int32),
                probs=torch.ones_like(context_metadata["topk_weights"]),
                restore_shape=self.hidden_shape_before_permute,
            )
            output = output.view(self.hidden_shape)
            return output

        TokenDispatcherWithAll2AllV._combine_postprocess = _patched_combine_postprocess
        TokenDispatcherWithAll2AllV._original_combine_postprocess = original_combine_postprocess

    except Exception as e:
        logger.warning("Failed to patch TokenDispatcherWithAll2AllV: %s", e)
