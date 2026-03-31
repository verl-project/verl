import logging
import os

import torch

logger = logging.getLogger(__name__)


def scaled_int8_per_channel(
    data_hp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight tensor to INT8 with per-channel (per-output-channel) scaling.

    This implements symmetric per-channel quantization suitable for Ascend W8A8_DYNAMIC:
        quantized = clamp(round(data / scale), -128, 127)
    where scale = max(abs(data_per_channel)) / 127

    Args:
        data_hp: Input high-precision tensor of shape (output_size, input_size).

    Returns:
        Tuple of (int8_data, weight_scale, weight_offset):
            - int8_data: INT8 quantized tensor of shape (output_size, input_size)
            - weight_scale: Per-channel scale of shape (output_size, 1), same dtype as input
            - weight_offset: Per-channel offset of shape (output_size, 1), same dtype as input
                            (zero for symmetric quantization)
    """
    assert len(data_hp.shape) == 2, f"Only 2D tensors supported, got shape {data_hp.shape}"

    original_dtype = data_hp.dtype
    data_fp32 = data_hp.float()

    max_abs = data_fp32.abs().amax(dim=1, keepdim=True)
    max_abs = torch.clamp(max_abs, min=1e-10)

    scale = max_abs / 127.0

    data_scaled = data_fp32 / scale
    data_scaled = data_scaled.round().clamp(-128, 127)

    int8_data = data_scaled.to(torch.int8)

    weight_scale = scale.to(original_dtype)
    weight_offset = torch.zeros_like(weight_scale)

    del data_fp32, data_scaled, max_abs, scale

    return int8_data, weight_scale, weight_offset
