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
import logging

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks
from torchtitan.models.attention import VarlenMetadata, create_attention_mask, get_causal_mask_mod
from torchtitan.protocols.model import AttentionMasksType

logger = logging.getLogger(__name__)

# Mapping from HuggingFace model_type to torchtitan model name.
# Torchtitan models not mapped here:
#   - flux: diffusion model, not applicable to verl's RL/SFT workflows
#   - llama3_ft: fault-tolerant variant of llama3, same HF models (mapped via "llama")
_HF_MODEL_TYPE_TO_TORCHTITAN_NAME = {
    "qwen2": "qwen3",
    "qwen3": "qwen3",
    "qwen2_moe": "qwen3",
    "qwen3_moe": "qwen3",
    "llama": "llama3",
    "llama4": "llama4",
    "deepseek_v3": "deepseek_v3",
    "gpt_oss": "gpt_oss",
}


def derive_torchtitan_name_and_flavor(hf_config) -> tuple[str, str]:
    """Derive torchtitan model name and flavor from a HuggingFace config.

    The name is mapped from ``hf_config.model_type``. The flavor is found by
    matching architecture parameters (dim, n_layers, vocab_size) against the
    known flavors registered in the torchtitan TrainSpec.

    Args:
        hf_config: A HuggingFace AutoConfig object.

    Returns:
        A ``(name, flavor)`` tuple.

    Raises:
        ValueError: If model_type is unsupported or no matching flavor is found.
    """
    import torchtitan.protocols.train_spec as train_spec_module

    model_type = getattr(hf_config, "model_type", None)
    if model_type is None:
        raise ValueError("HuggingFace config does not have 'model_type' field")

    name = _HF_MODEL_TYPE_TO_TORCHTITAN_NAME.get(model_type)
    if name is None:
        raise ValueError(
            f"Cannot derive torchtitan model name from HF model_type '{model_type}'. "
            f"Supported types: {list(_HF_MODEL_TYPE_TO_TORCHTITAN_NAME.keys())}."
        )

    train_spec = train_spec_module.get_train_spec(name)

    hidden_size = hf_config.hidden_size
    num_layers = hf_config.num_hidden_layers
    vocab_size = hf_config.vocab_size

    for flavor_name, model_args in train_spec.model_args.items():
        if (
            getattr(model_args, "dim", None) == hidden_size
            and getattr(model_args, "n_layers", None) == num_layers
            and getattr(model_args, "vocab_size", None) == vocab_size
        ):
            logger.info(
                f"Auto-derived torchtitan name='{name}', flavor='{flavor_name}' from HF model_type='{model_type}'"
            )
            return name, flavor_name

    raise ValueError(
        f"No matching torchtitan flavor found for model_type='{model_type}' "
        f"(hidden_size={hidden_size}, num_hidden_layers={num_layers}, "
        f"vocab_size={vocab_size}). "
        f"Available flavors for '{name}': {list(train_spec.model_args.keys())}."
    )


def enable_fsdp_gradient_division(model: nn.Module, dp_size: int) -> None:
    """
    Re-enable FSDP's automatic gradient division.

    TorchTitan calls disable_fsdp_gradient_division() which sets gradient_divide_factor=1.0.
    This re-enables it by setting the factor to the specified dp_size, so gradients are
    averaged across FSDP ranks. This is needed for verl's loss scaling (loss * dp_size)
    to work correctly.

    Args:
        model: The model (or model part) to enable gradient division on.
        dp_size: The data parallel size to use as the gradient divide factor.
    """

    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(float(dp_size))


def get_attention_masks(
    input_batch: torch.Tensor,
    positions: torch.Tensor,
    attn_type: str,
) -> AttentionMasksType:
    match attn_type:
        case "flex":
            return _get_flex_attention_masks(
                input_batch,
                positions,
            )
        case "varlen":
            return _create_varlen_metadata_for_document(
                input_batch,
                positions,
            )
        case _:
            raise TypeError("Only varlen and flex attn masks are supported")


def _get_document_mask_mod(positions: torch.Tensor) -> _mask_mod_signature:
    # Detect boundaries from position resets
    first_dummy_value = positions[:, :1] - 1
    position_diff = torch.diff(positions, prepend=first_dummy_value, dim=-1)
    sequence_indices = (position_diff != 1).cumsum(-1)  # [batch, seq]

    def document_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]

    return document_mask


def _get_flex_attention_masks(
    input_batch: torch.Tensor,
    positions: torch.Tensor,
) -> AttentionMasksType:
    mask_mods = [get_causal_mask_mod()]
    B = input_batch.shape[0]
    mask_mods.append(_get_document_mask_mod(positions=positions))
    return create_attention_mask(and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1])


def _create_varlen_metadata_for_document(input_batch: torch.Tensor, positions: torch.Tensor) -> VarlenMetadata:
    """
    Creates cumulative sequence length indices needed for variable length attention

    Args:
        input_batch: Input token IDs with shape [batch, seq].
        positions: Position IDs with shape [batch, seq]. Boundaries detected where
            position diff != 1 (i.e., position resets).

    Returns:
        VarlenMetadata containing cumulative sequence length indices for q, k, and max_seq_len
    """
    batch_size, seq_len = input_batch.shape
    device = input_batch.device

    # Detect boundaries from position resets (where diff != 1)
    first_dummy_value = positions[:, :1] - 1
    position_diff = torch.diff(positions, prepend=first_dummy_value, dim=-1)
    # boundary_mask[b, i] is True if position i starts a new document
    boundary_mask = position_diff != 1  # [batch, seq]
    boundary_mask[:, 0] = True

    cu_seqlens_list, all_seq_lengths = [], []
    offset = 0

    for b in range(batch_size):
        # Find positions where new documents start
        boundary_positions = boundary_mask[b].nonzero(as_tuple=True)[0].to(torch.int32)
        sample_cu_seqlens = torch.cat(
            [
                boundary_positions,
                torch.tensor([seq_len], dtype=torch.int32, device=device),
            ]
        )
        sample_cu_seqlens = torch.unique_consecutive(sample_cu_seqlens)

        seq_lengths = torch.diff(sample_cu_seqlens)
        all_seq_lengths.append(seq_lengths)

        cu_seqlens_adjusted = sample_cu_seqlens[:-1] + offset
        cu_seqlens_list.append(cu_seqlens_adjusted)

        offset += seq_len

    packed_cu_seqlens = torch.cat(cu_seqlens_list + [torch.tensor([offset], dtype=torch.int32, device=device)])

    max_seqlen = 0
    if len(all_seq_lengths) > 0:
        all_seq_lengths = torch.cat(all_seq_lengths)
        # device to host sync but only done once per model forward
        max_seqlen = all_seq_lengths.max().item()

    return VarlenMetadata(
        cu_seq_q=packed_cu_seqlens,
        cu_seq_k=packed_cu_seqlens,
        max_q=max_seqlen,
        max_k=max_seqlen,
    )
