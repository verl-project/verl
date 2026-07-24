# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""GPU smoke: verl wiring -> Megatron-Core TensorParallelMuon, real build + one step.

This is NOT a pytest-collected CPU test (Megatron's DDP grad buffers allocate on
``torch.cuda.current_device()``, so the full optimizer object can only be built on a GPU).
Run it under torchrun on a single GPU inside a container that has Megatron-Core with
``emerging_optimizers`` and verl installed::

    torchrun --nproc_per_node=1 tests/utils/megatron/muon_optim_gpu_smoke.py

Success prints ``MUON_GPU_SMOKE_OK`` and exits 0. It proves verl's
``init_megatron_optim_config`` + ``get_megatron_optimizer`` actually construct Megatron's
emerging (Muon) optimizer -- a ChainedOptimizer containing a ``TensorParallelMuon`` -- rather
than silently falling back to Adam, and that one ``optimizer.step()`` updates parameters.
"""

import os

import torch
import torch.nn as nn


def main():
    assert torch.cuda.is_available(), "this smoke requires a GPU"
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon  # noqa: F401
    from megatron.core.transformer import TransformerConfig

    from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config
    from verl.workers.config.optimizer import McoreOptimizerConfig

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(80, 48)
            self.fc2 = nn.Linear(48, 16)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    torch.manual_seed(0)
    model = Net().bfloat16().cuda()
    model.requires_grad_(True)

    tconf = TransformerConfig(num_attention_heads=1, num_layers=1)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    ddp_model = DistributedDataParallel(tconf, ddp_config, model)

    vcfg = McoreOptimizerConfig(
        lr=0.01,
        optimizer="muon",
        weight_decay=0.01,
        clip_grad=1.0,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_num_ns_steps=5,
        muon_scale_mode="spectral",
        muon_tp_mode="duplicated",
    )
    mcore_cfg = init_megatron_optim_config(vcfg, use_distributed_optimizer=False, fp16=False)
    assert mcore_cfg.optimizer == "muon"
    print("[ok] verl init_megatron_optim_config -> OptimizerConfig(optimizer=muon)")

    optimizer = get_megatron_optimizer(ddp_model, mcore_cfg)
    assert optimizer is not None
    inner_types = [
        type(getattr(ch, "optimizer", ch)).__name__ for ch in getattr(optimizer, "chained_optimizers", [optimizer])
    ]
    print("[ok] get_megatron_optimizer -> inner optimizers:", inner_types)
    assert any("Muon" in t for t in inner_types), f"expected TensorParallelMuon, got {inner_types}"
    print("[ok] TensorParallelMuon constructed (NOT a silent Adam fallback)")

    x = torch.randn(16, 80, dtype=torch.bfloat16, device="cuda")
    loss = ddp_model(x).sum()
    loss.backward()
    before = {n: p.data.clone() for n, p in ddp_model.named_parameters()}
    optimizer.step()
    updated = sum(1 for n, p in ddp_model.named_parameters() if not torch.equal(p.data, before[n]))
    assert updated > 0, "no parameters updated after optimizer.step()"
    print(f"[ok] optimizer.step() updated {updated} parameter tensors")
    print("MUON_GPU_SMOKE_OK")


if __name__ == "__main__":
    main()
