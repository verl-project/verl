"""Use training AttnRes/RMSNorm arithmetic while preserving native residual state."""
import os
import torch
import torch_npu


def attn_res_forward(prefix_sum, addend, blocks, proj_weight, norm_weight, norm_eps,
                     valid_blocks, output_norm_weight=None, output_norm_eps=0.,
                     block_write_idx=-1, return_materialized=False, *, mix=True,
                     optimize_prefill=False):
    result = torch.ops._C_ascend.attn_res_fwd(
        prefix_sum, addend, blocks, proj_weight, norm_weight, norm_eps,
        valid_blocks, output_norm_weight, output_norm_eps, block_write_idx,
        return_materialized, mix=mix, optimize_prefill=optimize_prefill)
    if os.environ.get('KIMI_ALIGNMENT_NATIVE_ATTN_RES') != '1' or not mix:
        return result
    # Keep native BF16 addition, block writes and raw-prefix state exactly intact.
    raw_prefix = result[1]
    if valid_blocks:
        from cann_ops_transformer import block_attention_residuals
        mixed = block_attention_residuals(
            raw_prefix, blocks[:, :valid_blocks].contiguous(), proj_weight,
            norm_weight, norm_eps=norm_eps)
    else:
        mixed = raw_prefix
    normalized = mixed if output_norm_weight is None else torch_npu.npu_rms_norm(
        mixed, output_norm_weight, epsilon=output_norm_eps)[0]
    return normalized, raw_prefix, mixed if return_materialized else result[2]
