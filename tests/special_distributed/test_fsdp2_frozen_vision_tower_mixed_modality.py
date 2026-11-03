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

"""Mixed-modality FSDP2 training with a frozen vision tower, on 2 CPU ranks.

With ``freeze_vision_tower=True`` the tower is excluded from per-module FSDP2
wrapping, so its params join the root FSDP unit. Root-unit params are
all-gathered uniformly on every rank at model forward entry, which removes the
cross-rank vision-tower entry-count alignment requirement (#3491/#3496) — a
rank holding an image batch runs the (frozen) tower while a text-only rank
skips it entirely, without dummy visual forwards and without hanging.

Verifies: forward/backward completes on both ranks, vision params receive no
grads, language params do, and checkpoint keys are unchanged by the exclusion.

Launch:
    torchrun --nproc-per-node=2 --standalone \\
        tests/special_distributed/test_fsdp2_frozen_vision_tower_mixed_modality.py
"""

from types import SimpleNamespace

import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import apply_fsdp2

VOCAB_SIZE = 64
HIDDEN_SIZE = 32
IMAGE_TOKEN_ID = 1
N_IMAGE_TOKENS = 4


class _VisionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _VisionTower(torch.nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.patch_embed = torch.nn.Linear(12, HIDDEN_SIZE)
        self.blocks = torch.nn.ModuleList([_VisionBlock() for _ in range(depth)])

    def forward(self, pixel_values, grid_thw=None):
        x = self.patch_embed(pixel_values)
        for block in self.blocks:
            x = block(x)
        return x


class _DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.mlp = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, x):
        return x + self.mlp(F.gelu(self.self_attn(x)))


class _MockVLM(torch.nn.Module):
    _no_split_modules = ["_DecoderLayer", "_VisionBlock"]

    def __init__(self, num_layers=2):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False, image_token_id=IMAGE_TOKEN_ID)
        self.visual = _VisionTower()
        self.embed_tokens = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = torch.nn.ModuleList([_DecoderLayer() for _ in range(num_layers)])
        self.lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, input_ids, pixel_values=None):
        embeds = self.embed_tokens(input_ids)
        if pixel_values is not None:
            # Real image path: run the (frozen) tower and scatter into image positions.
            image_embeds = self.visual(pixel_values)
            mask = input_ids == IMAGE_TOKEN_ID
            embeds = embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds), image_embeds.to(embeds.dtype))
        h = embeds
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)


def _make_text_batch():
    return torch.randint(2, VOCAB_SIZE, (2, 8))


def _make_image_batch():
    """Text ids with 4 image-token positions; patches shaped to match them."""
    ids = torch.randint(2, VOCAB_SIZE, (2, 8))
    ids[:, 2 : 2 + N_IMAGE_TOKENS] = IMAGE_TOKEN_ID
    pixel_values = torch.randn(2 * N_IMAGE_TOKENS, 12)
    return ids, pixel_values


def main():
    initialize_global_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert world_size == 2, "this test needs exactly 2 ranks"
    device_mesh = init_device_mesh("cpu", (world_size,))

    model = _MockVLM()
    expected_keys = set(model.state_dict().keys())

    # freeze_vision_tower=True, as FSDPEngine._freeze_vision_tower applies it.
    model.visual.requires_grad_(False)

    fsdp_kwargs = {
        "mesh": device_mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
    }
    apply_fsdp2(model, fsdp_kwargs, {}, exclude_modules=[model.visual])

    # Checkpoint compatibility: the exclusion must not change state_dict keys.
    assert set(model.state_dict().keys()) == expected_keys, "state_dict keys changed by wrap exclusion"

    # Mixed modality: rank 0 holds an image batch, rank 1 a text-only batch.
    if rank == 0:
        ids, pixel_values = _make_image_batch()
        logits = model(ids, pixel_values=pixel_values)
    else:
        ids = _make_text_batch()
        logits = model(ids)
    loss = logits.float().pow(2).mean()
    loss.backward()

    # Frozen tower: no grads anywhere inside it.
    for name, p in model.visual.named_parameters():
        assert p.grad is None, f"frozen vision param {name} unexpectedly received a grad"

    # Language side: grads exist (as DTensors on the sharded params).
    lm_param_names = [n for n, p in model.named_parameters() if not n.startswith("visual.") and p.grad is not None]
    assert lm_param_names, "language params received no grads"

    torch.distributed.barrier()
    if rank == 0:
        print("PASS: mixed-modality FSDP2 with frozen vision tower (2 ranks)")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
