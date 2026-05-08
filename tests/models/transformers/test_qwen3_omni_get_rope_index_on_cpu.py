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
"""CPU regression tests for ``verl.models.transformers.qwen3_omni_moe.get_rope_index``.

HF's ``Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index`` gates the
3D audio/image/video-aware branch on ``image_grid_thw is not None or
video_grid_thw is not None``. Audio-only samples fall through to a 1D
``cumsum`` broadcast to 3 axes in ``float32``. That makes the FSDP2 actor
disagree with the vLLM rollout engine on audio-region tokens.

These tests pin the verl helper's contract so the bug can't silently return.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from verl.models.transformers.qwen3_omni_moe import get_rope_index

AUDIO_TOKEN_ID = 151675
AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670


class _FakeQwen3OmniProcessor:
    """Minimal stand-in for ``Qwen3OmniMoeProcessor`` — only the fields the
    audio-aware branch of ``get_rope_index`` reads from.

    ``get_rope_index`` on the real processor is dynamically bound by
    ``verl.utils.tokenizer.hf_processor``; we don't need it here because the
    verl helper's audio-only branch is self-contained (no delegation).
    """

    __class__name = "Qwen3OmniMoeProcessor"

    def __init__(self):
        self.config = SimpleNamespace(
            audio_token_id=AUDIO_TOKEN_ID,
            audio_start_token_id=AUDIO_START_TOKEN_ID,
            audio_end_token_id=AUDIO_END_TOKEN_ID,
        )
        self.audio_token_id = AUDIO_TOKEN_ID
        self.audio_start_token_id = AUDIO_START_TOKEN_ID
        self.audio_end_token_id = AUDIO_END_TOKEN_ID


def _build_row(audio_lens: list[int]):
    """Build input_ids = [t t t] + (audio_start + audio_pad*L + audio_end + [t])*len + tail.

    Audio length L is the post-Whisper token count produced by
    ``_get_feat_extract_output_lengths``. For ``audio_seqlens = 1104`` the
    formula gives ``L = 144``; for 1204 it gives 157; for 2002 it gives 261.
    We pick audio_seqlens values that return the caller-supplied L.
    """
    # Inverse of _get_feat_extract_output_lengths for multiples of 100:
    # L = 13 * (seq // 100)  when seq % 100 == 0
    # so pass seq = L * 100 / 13 rounded up. Simpler: pick a fixed seq that
    # we know produces L (see test_lengths fixture below).
    pass  # helper removed — we inline the cases below


def _feat_extract(seq):
    """Mirror of transformers._get_feat_extract_output_lengths for audit."""
    input_lengths_leave = seq % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (seq // 100) * 13


def test_audio_only_returns_long_not_float():
    """Pre-fix the audio-only path returned ``float32``; after fix it must be
    ``int64`` so HF rope embeddings (indexed by position) don't silently cast."""
    processor = _FakeQwen3OmniProcessor()
    audio_len = _feat_extract(1104)  # 144

    row = [10, 11, 12, AUDIO_START_TOKEN_ID] + [AUDIO_TOKEN_ID] * audio_len + [AUDIO_END_TOKEN_ID, 20]
    input_ids = torch.tensor([row], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    audio_seqlens = torch.tensor([1104], dtype=torch.long)

    pid, deltas = get_rope_index(
        processor,
        input_ids=input_ids,
        attention_mask=attn,
        audio_seqlens=audio_seqlens,
    )

    assert pid.dtype == torch.long
    assert list(pid.shape) == [3, 1, input_ids.shape[-1]]


def test_audio_only_positions_are_contiguous_and_cover_audio_block():
    """The verl helper must emit ``[0, 1, ..., T-1]`` for a single-audio row
    with no other text gaps — that's what HF's audio-aware branch does via
    ``torch.arange(audio_len).view(1, -1).expand(3, -1)`` with ``st_idx``
    bookkeeping. The pre-fix code produced the same *values* but as float32,
    so we fix the dtype and *also* check the arithmetic stays right."""
    processor = _FakeQwen3OmniProcessor()
    audio_len = _feat_extract(1104)

    row = [10, 11, 12, 13, 14, AUDIO_START_TOKEN_ID] + [AUDIO_TOKEN_ID] * audio_len + [AUDIO_END_TOKEN_ID, 20, 21, 22]
    T = len(row)
    input_ids = torch.tensor([row], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    audio_seqlens = torch.tensor([1104], dtype=torch.long)

    pid, _ = get_rope_index(
        processor,
        input_ids=input_ids,
        attention_mask=attn,
        audio_seqlens=audio_seqlens,
    )

    expected = torch.arange(T, dtype=torch.long)
    assert torch.equal(pid[0, 0], expected)
    assert torch.equal(pid[1, 0], expected)
    assert torch.equal(pid[2, 0], expected)


def test_two_audio_blocks_positions_cover_full_sequence():
    processor = _FakeQwen3OmniProcessor()
    audio_len = _feat_extract(1104)

    block = [AUDIO_START_TOKEN_ID] + [AUDIO_TOKEN_ID] * audio_len + [AUDIO_END_TOKEN_ID]
    row = [10, 11] + block + [30, 31] + block + [40]
    T = len(row)
    input_ids = torch.tensor([row], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    audio_seqlens = torch.tensor([1104, 1104], dtype=torch.long)

    pid, _ = get_rope_index(
        processor,
        input_ids=input_ids,
        attention_mask=attn,
        audio_seqlens=audio_seqlens,
    )
    expected = torch.arange(T, dtype=torch.long)
    assert torch.equal(pid[0, 0], expected)


def test_no_audio_no_vision_produces_long_dtype():
    """Pure-text branch delegates to HF and casts to long. Uses a stubbed
    ``get_rope_index`` on the processor since the verl helper only falls
    through when ``audio_seqlens is None`` and no vision grids."""

    class _Stub(_FakeQwen3OmniProcessor):
        def get_rope_index(self, **kwargs):
            T = kwargs["input_ids"].shape[-1]
            # Mimic HF's text-only fallback: float cumsum broadcast to 3 axes.
            pos = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(3, -1, -1)
            return pos, torch.tensor([[0]])

    processor = _Stub()
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    attn = torch.ones_like(input_ids)

    pid, _ = get_rope_index(processor, input_ids=input_ids, attention_mask=attn)
    assert pid.dtype == torch.long
