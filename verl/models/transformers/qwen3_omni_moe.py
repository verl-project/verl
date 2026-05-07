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
"""Monkey patches for the Qwen3-Omni MoE model family.

Upstream ``Qwen3OmniMoeThinkerTextExperts.forward`` (and the Talker
variant) can dispatch through different expert implementations depending on
``config._experts_implementation``. The training monkey patch routes the real
Qwen3-Omni model type here so configs that still have that value unset can opt
into the configured implementation, while user- or transformers-provided
values are left untouched.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Matches the keys registered in transformers.integrations.moe.ExpertsInterface.
_DEFAULT_EXPERTS_IMPLEMENTATION = "batched_mm"


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    audio_seqlens: Optional[torch.Tensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Audio-aware MRope for Qwen3-Omni.

    HF's ``Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index`` gates
    the 3D audio/image/video-aware branch on ``image_grid_thw is not None or
    video_grid_thw is not None``. Audio-only samples fall through to a 1D
    ``cumsum`` broadcast across 3 axes, which corrupts MRope at the audio
    region and makes actor logprobs diverge from the rollout engine.

    This helper reconstructs the audio-aware branch locally: when
    ``audio_seqlens`` is present it walks the pre-expanded ``input_ids`` and
    assigns per-axis MRope positions around ``audio_start / audio_pad*N /
    audio_end`` blocks. When images/videos are present, it delegates back to
    the processor's bound ``get_rope_index`` (which works correctly once at
    least one vision grid is supplied, provided ``get_llm_pos_ids_for_vision``
    is also bound by ``verl.utils.tokenizer.hf_processor``).

    Returns ``(position_ids, mrope_position_deltas)`` matching HF's
    signature: ``position_ids`` has shape ``(3, batch, seq_len)`` and
    ``dtype=torch.long``. Callers that feed the result into
    ``Qwen3OmniMoeThinkerTextModel.forward`` must prepend a text axis
    (cumsum over the valid attention mask) to reach the
    ``(4, bs, seq_len)`` layout HF's text model expects — see
    ``verl.experimental.agent_loop.agent_loop.AgentLoopWorker._compute_position_ids``
    for the concat recipe.
    """

    def _assert_shape(pos: torch.Tensor) -> torch.Tensor:
        assert pos.dim() == 3 and pos.shape[0] == 3, (
            f"qwen3_omni get_rope_index must return (3, bs, seq); got {tuple(pos.shape)}"
        )
        return pos

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        _get_feat_extract_output_lengths,
    )

    has_vision = image_grid_thw is not None or video_grid_thw is not None
    has_audio = audio_seqlens is not None

    # Vision branch: HF's bound implementation already works. Fall back to
    # it when any vision grid is provided — the guard fires correctly.
    if has_vision:
        pos, deltas = processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_audio_in_video=use_audio_in_video,
            audio_seqlens=audio_seqlens,
            second_per_grids=second_per_grids,
        )
        return _assert_shape(pos), deltas

    # Pure-text branch: HF fallback is correct (the 1D linear cumsum is what
    # a text-only sample needs). Cast to long for consistency.
    if not has_audio:
        pos, deltas = processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=None,
            video_grid_thw=None,
            use_audio_in_video=use_audio_in_video,
            audio_seqlens=None,
            second_per_grids=second_per_grids,
        )
        return _assert_shape(pos.long()), deltas

    # Audio-only branch: replicate HF's audio-aware loop but without the
    # buggy guard. Mirrors transformers 4.57.1
    # ``Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index``.
    cfg = getattr(processor, "config", None)
    audio_token_id = getattr(processor, "audio_token_id", None)
    audio_start_token_id = getattr(processor, "audio_start_token_id", None)
    audio_end_token_id = getattr(processor, "audio_end_token_id", None)
    if cfg is not None:
        audio_token_id = getattr(cfg, "audio_token_id", audio_token_id)
        audio_start_token_id = getattr(cfg, "audio_start_token_id", audio_start_token_id)
        audio_end_token_id = getattr(cfg, "audio_end_token_id", audio_end_token_id)
    if audio_token_id is None or audio_start_token_id is None or audio_end_token_id is None:
        raise RuntimeError(
            "Qwen3-Omni audio MRope requires audio_token_id / audio_start_token_id / "
            "audio_end_token_id on the processor; did you forget ensure_qwen3_omni_processor_attrs?"
        )

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    batch_size, seq_len = input_ids.shape
    position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.long, device=input_ids.device)
    mrope_position_deltas = []

    attn_bool = attention_mask == 1

    audio_idx_global = 0
    for i in range(batch_size):
        row_ids = input_ids[i][attn_bool[i]]
        row_tokens = row_ids.tolist()
        row_len = len(row_tokens)

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        remain_audios = int((row_ids == audio_start_token_id).sum())
        bos_len, eos_len = 1, 1

        for _ in range(remain_audios):
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            try:
                ed_audio_start = row_tokens.index(audio_start_token_id, st)
            except ValueError:
                break

            # Text run before this audio block.
            text_len = ed_audio_start - st
            if text_len > 0:
                llm_pos_ids_list.append(torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)
                st_idx += text_len

            # audio_start (bos_len=1).
            llm_pos_ids_list.append(torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)
            st_idx += bos_len

            # audio_pad*N_audio (N_audio from _get_feat_extract_output_lengths).
            audio_len_t = _get_feat_extract_output_lengths(audio_seqlens[audio_idx_global])
            audio_len = int(audio_len_t.item() if torch.is_tensor(audio_len_t) else audio_len_t)
            llm_pos_ids_list.append(torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)

            # advance st past text + audio_start + audio_pad*N + audio_end.
            st += int(text_len + bos_len + audio_len + eos_len)
            audio_idx_global += 1

            # audio_end (eos_len=1).
            st_idx = int(llm_pos_ids_list[-1].max()) + 1
            llm_pos_ids_list.append(torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)

        # Trailing text run.
        if st < row_len:
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            tail_len = row_len - st
            llm_pos_ids_list.append(torch.arange(tail_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)
        elif not llm_pos_ids_list:
            # Attention-masked-to-zero or empty row.
            llm_pos_ids_list.append(torch.zeros(3, row_len, dtype=torch.long))

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attn_bool[i]] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(int(llm_positions.max()) + 1 - row_len)

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return _assert_shape(position_ids), mrope_position_deltas


def _force_experts_implementation(config, implementation: str) -> str:
    """Set ``config._experts_implementation`` if it is not already configured.

    ``PreTrainedConfig._experts_implementation`` has a setter that walks
    ``sub_configs`` for us, so assigning on the outer composite config is
    enough — both ``thinker_config`` and ``talker_config`` pick up the value.
    If the user or transformers has already set it explicitly, we leave it
    alone. In particular, forcing HF ``batched_mm`` on Qwen3-Omni long
    sequences can materialize huge selected-weight tensors during actor
    logprob forward.
    """
    current = getattr(config, "_experts_implementation", None)
    if current is not None:
        logger.debug(
            "Skipping _experts_implementation override on %s (already set to %r).",
            type(config).__name__,
            current,
        )
        return current
    config._experts_implementation = implementation

    return getattr(config, "_experts_implementation", implementation)


def patch_qwen3_omni_moe_sparse_moe_block_forward(
    model: Optional[torch.nn.Module] = None,
    implementation: str = _DEFAULT_EXPERTS_IMPLEMENTATION,
) -> None:
    """Configure Qwen3-Omni expert dispatch when it is still unset.

    Both ``Qwen3OmniMoeThinkerTextExperts`` and ``Qwen3OmniMoeTalkerTextExperts``
    already carry ``@use_experts_implementation`` in transformers >=5.3, so the
    only thing we need is to flip ``config._experts_implementation`` away from
    its ``None`` default. Already-instantiated models can carry a concrete
    value such as ``eager``; this helper does not override those values.
    """
    if model is None:
        logger.debug(
            "Qwen3-Omni experts dispatch requires a model instance to set "
            "config._experts_implementation; skipping (no-op)."
        )
        return

    config = getattr(model, "config", None)
    if config is None:
        logger.warning("Qwen3-Omni experts dispatch: model has no .config attribute; skipping.")
        return

    applied_implementation = _force_experts_implementation(config, implementation)
    logger.info(
        "Qwen3-Omni config._experts_implementation=%s after sparse MoE patch.",
        applied_implementation,
    )
