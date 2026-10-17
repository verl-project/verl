# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""CPU-only tests for the frozen-vision-tower guard in the VLM adapters.

Every multimodal adapter runs a dummy visual forward on text-only batches so
that vision params stay in the autograd graph and every rank enters the vision
tower the same number of times under FSDP (#3496). The forward may only be
skipped for a tower that is fully frozen, holds no trainable params, AND is
kept out of per-module FSDP wrapping — FSDPEngine marks such towers with
``_verl_frozen_vision_tower``. ``requires_grad`` alone must never trigger the
skip: PEFT clears it on the whole non-adapter model for every LoRA run, while
the tower stays per-module wrapped (#3491 alignment). And trainable params
inside the tower (vision LoRA, QAT) must keep the dummy: the root FSDP unit
cannot take per-rank-asymmetric gradients (FSDP1 hangs). These tests pin all
of that with a mock tower, for all four adapter families.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from verl.models.transformers import glm4v, qwen2_vl, qwen3_5, qwen3_vl

ADAPTERS = {
    "qwen2_vl": qwen2_vl,
    "qwen3_vl": qwen3_vl,
    "glm4v": glm4v,
    "qwen3_5": qwen3_5,
}

VOCAB_SIZE = 64
HIDDEN_SIZE = 32


class _VisualOutput:
    """Mimics the transformers>=5 visual tower output: `.pooler_output` tensor."""

    def __init__(self, tensor):
        self.pooler_output = tensor


class _MockVisionTower(nn.Module):
    # 1176 = 3 (in_channels) * 2 (temporal_patch_size) * 14^2 (patch_size),
    # matching the dummy patch_dim the adapters compute from vision_config
    # (glm4v hardcodes the same value).
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(1176, HIDDEN_SIZE)
        self.calls = 0

    def forward(self, pixel_values, grid_thw=None):
        self.calls += 1
        return _VisualOutput(self.patch_embed(pixel_values))


class _MockVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _MockVisionTower()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.config = SimpleNamespace(
            image_token_id=1,
            video_token_id=2,
            vision_config=SimpleNamespace(in_channels=3, temporal_patch_size=2, patch_size=14),
        )

    def get_input_embeddings(self):
        return self.embed_tokens


def _text_only_embeds(adapter, model, input_ids):
    out = adapter._get_input_embeds(model, input_ids, attention_mask=torch.ones_like(input_ids))
    if isinstance(out, dict):
        return out["inputs_embeds"]
    return out[0]


@pytest.mark.parametrize("family", sorted(ADAPTERS))
def test_unfrozen_tower_keeps_dummy_visual_forward(family):
    """Default behavior is unchanged: the dummy forward runs exactly once (#3496)."""
    adapter = ADAPTERS[family]
    model = _MockVLM()
    input_ids = torch.randint(3, VOCAB_SIZE, (2, 7))

    embeds = _text_only_embeds(adapter, model, input_ids)

    assert model.visual.calls == 1, "dummy visual forward must run exactly once when the tower trains"
    torch.testing.assert_close(embeds, model.embed_tokens(input_ids))


@pytest.mark.parametrize("family", sorted(ADAPTERS))
def test_marked_frozen_tower_skips_dummy_visual_forward(family):
    """With freeze_vision_tower applied (frozen + wrap-excluded + marked),
    text-only batches never enter the tower."""
    adapter = ADAPTERS[family]
    model = _MockVLM()
    model.visual.requires_grad_(False)
    model.visual._verl_frozen_vision_tower = True
    input_ids = torch.randint(3, VOCAB_SIZE, (2, 7))

    embeds = _text_only_embeds(adapter, model, input_ids)

    assert model.visual.calls == 0, "dummy visual forward must be skipped for a marked frozen tower"
    torch.testing.assert_close(embeds, model.embed_tokens(input_ids))


@pytest.mark.parametrize("family", sorted(ADAPTERS))
def test_peft_frozen_tower_without_marker_keeps_dummy_visual_forward(family):
    """requires_grad=False alone must NOT skip: a language-targeted LoRA run
    freezes the whole non-adapter model (PEFT) while the tower stays
    per-module wrapped — skipping there deadlocks mixed-modality batches."""
    adapter = ADAPTERS[family]
    model = _MockVLM()
    model.visual.requires_grad_(False)  # PEFT did this; no marker set
    input_ids = torch.randint(3, VOCAB_SIZE, (2, 7))

    _text_only_embeds(adapter, model, input_ids)

    assert model.visual.calls == 1, "dummy visual forward must run without the engine's marker"


@pytest.mark.parametrize("family", sorted(ADAPTERS))
def test_marked_frozen_tower_with_trainable_params_keeps_dummy_visual_forward(family):
    """A trainable param inside the tower (vision LoRA, QAT linears) must keep
    the dummy even if the marker is set: on mixed-modality batches the root FSDP
    unit would receive gradients on image ranks only, which FSDP1's root-unit
    gradient reduce does not support (verified to hang). The engine never sets
    the marker in this state; this pins the adapters' defense in depth."""
    adapter = ADAPTERS[family]
    model = _MockVLM()
    model.visual.requires_grad_(False)
    model.visual._verl_frozen_vision_tower = True
    # Simulate a PEFT adapter: one trainable submodule inside the frozen tower.
    model.visual.lora_A = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    input_ids = torch.randint(3, VOCAB_SIZE, (2, 7))

    _text_only_embeds(adapter, model, input_ids)

    assert model.visual.calls == 1, "trainable params inside the tower must keep the dummy forward"


@pytest.mark.parametrize("family", sorted(ADAPTERS))
def test_marked_frozen_tower_with_trainable_sibling_keeps_dummy_visual_forward(family):
    """A trainable visual-path SIBLING (e.g. a projector whose training is
    re-enabled after the engine marked the path) must keep the dummy exactly
    like a trainable param inside the tower: the shared gate scans the whole
    visual path, not just model.visual — a trainable sibling in the root FSDP
    unit would receive gradients on image ranks only (the FSDP1 hang)."""
    adapter = ADAPTERS[family]
    model = _MockVLM()
    model.visual.requires_grad_(False)
    model.visual._verl_frozen_vision_tower = True
    model.multi_modal_projector = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)  # trainable sibling
    input_ids = torch.randint(3, VOCAB_SIZE, (2, 7))

    _text_only_embeds(adapter, model, input_ids)

    assert model.visual.calls == 1, "a trainable visual-path sibling must keep the dummy forward"
