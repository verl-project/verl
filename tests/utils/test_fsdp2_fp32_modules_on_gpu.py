import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 mixed precision needs a GPU")


class MockVisionBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)

    def forward(self, x):
        return x + self.fc(x)


class MockVision(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        self.patch = nn.Linear(hidden, hidden)
        self.pos_embed = nn.Embedding(4, hidden)
        self.blocks = nn.ModuleList([MockVisionBlock(hidden) for _ in range(2)])

    def forward(self, pixels):
        x = self.patch(pixels) + self.pos_embed(torch.arange(4, device=pixels.device)).sum(0)
        for block in self.blocks:
            x = block(x)
        return x


class MockDecoderLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)

    def forward(self, x):
        return x + self.fc(x)


class MockVLM(nn.Module):
    _no_split_modules = ["MockDecoderLayer", "MockVisionBlock"]

    def __init__(self, hidden=64):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.model = nn.Module()
        self.model.visual = MockVision(hidden)
        self.model.layers = nn.ModuleList([MockDecoderLayer(hidden) for _ in range(2)])
        self.lm_head = nn.Linear(hidden, 8, bias=False)

    def forward(self, pixels):
        x = self.model.visual(pixels).to(torch.bfloat16)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


@pytest.fixture(scope="module")
def mesh():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    yield init_device_mesh("cuda", (1,))
    dist.destroy_process_group()


@pytest.mark.parametrize("tf32", [False, True])
def test_fp32_modules_compute_in_fp32_and_the_rest_in_bf16(mesh, tf32):
    torch.manual_seed(0)
    model = MockVLM().cuda()
    seen = {}

    def record(name):
        # input, weight and output dtype of a linear layer: under autocast the output shows what the matmul ran in
        def hook(module, args, out):
            seen[name] = (args[0].dtype, module.weight.dtype, out.dtype)

        return hook

    model.model.visual.blocks[0].fc.register_forward_hook(record("vision block"))
    model.model.visual.patch.register_forward_hook(record("vision patch"))
    model.model.layers[0].fc.register_forward_hook(record("decoder layer"))
    fsdp_kwargs = {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
    }

    apply_fsdp2(model, fsdp_kwargs, {"mixed_precision": {"fp32_modules": ["model.visual"], "fp32_modules_tf32": tf32}})
    # the engine runs its forward under autocast in the param dtype (FSDPEngine's autocast_ctx)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(torch.randn(3, 64, device="cuda", dtype=torch.bfloat16))
    out.float().sum().backward()

    # the vision tower's own parameters (patch, pos_embed) and its blocks gather and compute in fp32
    assert seen["vision patch"] == (torch.float32, torch.float32, torch.float32)
    assert seen["vision block"] == (torch.float32, torch.float32, torch.float32)
    assert seen["decoder layer"] == (torch.bfloat16, torch.bfloat16, torch.bfloat16)
    assert model.model.visual.config._attn_implementation == "sdpa"
    assert torch.backends.cuda.matmul.allow_tf32 == tf32 and torch.backends.cudnn.allow_tf32 == tf32
    assert all(p.grad is not None for p in model.model.visual.parameters())
