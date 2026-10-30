# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Exercise configured precision through native FSDP forward/backward and updates."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _TinyModel(torch.nn.Module):
    _no_split_modules = ["Linear"]

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 4, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def forward(self, x, use_cache=False):
        return self.proj(x)


@pytest.fixture(scope="module")
def device_mesh(tmp_path_factory):
    torch.cuda.set_device(0)
    rendezvous = tmp_path_factory.mktemp("fsdp_dtype") / "rendezvous"
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cuda", (1,), mesh_dim_names=("fsdp",))
    finally:
        dist.destroy_process_group()


def _snapshot(module):
    return [
        (param.to_local() if isinstance(param, DTensor) else param).detach().clone() for param in module.parameters()
    ]


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
@pytest.mark.parametrize(
    "dtype,mixed_precision,expected",
    [
        ("bfloat16", None, torch.bfloat16),
        ("float16", None, torch.float16),
        ("float32", None, torch.float32),
        ("float16", {"reduce_dtype": "fp32"}, torch.float16),
        ("float16", {"param_dtype": "bf16"}, torch.bfloat16),
        ("bfloat16", {"param_dtype": "fp16"}, torch.float16),
    ],
)
def test_fsdp_dtype_training_step(device_mesh, strategy, dtype, mixed_precision, expected):
    torch.manual_seed(42)
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.engine_config = FSDPEngineConfig(strategy=strategy, dtype=dtype, mixed_precision=mixed_precision)
    engine.model_config = SimpleNamespace(lora_rank=0, enable_activation_offload=False)
    engine.device_mesh = device_mesh
    engine.optimizer_config = FSDPOptimizerConfig(clip_grad=1.0)
    engine._qat_enabled = False
    engine.module = engine._build_fsdp_module(_TinyModel().cuda())
    engine.optimizer = torch.optim.SGD(engine.module.parameters(), lr=0.01)
    engine.prepare_model_inputs = lambda micro_batch: ({"x": micro_batch["x"]}, {})
    engine.prepare_model_outputs = lambda output, **kwargs: {"projection": output}
    engine.get_data_parallel_group = lambda: dist.group.WORLD

    before = _snapshot(engine.module)
    batch = TensorDict({"x": torch.full((2, 8), 0.125, device="cuda")}, batch_size=[2])

    def loss_function(model_output, data, dp_group):
        # Keep the loss small enough for the default fp16 loss scale.
        return model_output["projection"].float().square().mean(), {}

    for _ in range(2):
        loss, output = engine.forward_step(batch, loss_function, forward_only=False)
        assert output["model_output"]["projection"].dtype == expected
        assert engine._autocast_dtype == expected
        assert (engine.scaler is not None) == (expected == torch.float16)
        if engine.scaler is not None:
            assert engine.scaler.is_enabled()
            engine.scaler.scale(loss).backward()
        else:
            loss.backward()

    grad_norm = engine.optimizer_step()
    assert math.isfinite(grad_norm) and grad_norm > 0
    assert any(not torch.equal(old, new) for old, new in zip(before, _snapshot(engine.module), strict=True))
