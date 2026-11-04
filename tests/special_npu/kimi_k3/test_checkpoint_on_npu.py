# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# ruff: noqa: E402

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

pytest.importorskip("torch_npu")

from fsdp_turbo.models.kimi.kimi_k3.checkpoint import load_hf_checkpoint_shards

from verl.utils.device import get_torch_device
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _check_sharded_meta_load(rank, directory):
    get_torch_device().set_device(rank)
    dist.init_process_group("cpu:gloo,npu:hccl", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2)
    try:
        mesh = DeviceMesh("npu", [0, 1])
        model = torch.nn.Linear(6, 4, bias=False, device="meta")
        model.weight = torch.nn.Parameter(distribute_tensor(model.weight, mesh, [Shard(0)]))
        original_parameter = model.weight
        model.register_buffer("rotary", torch.arange(3).float(), persistent=False)
        load_hf_checkpoint_shards(model, directory, materialize_device=f"npu:{rank}")
        assert model.weight is original_parameter
        assert model.weight.placements == (Shard(0),)
        assert model.weight.device_mesh == mesh
        assert model.weight.shape == (4, 6)
        expected = torch.arange(24).reshape(4, 6).float()[rank * 2 : rank * 2 + 2]
        assert model.weight.to_local().device.type == "npu"
        torch.testing.assert_close(model.weight.to_local().cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(model.rotary.cpu(), torch.arange(3).float(), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_streaming_load_materializes_dtensor_with_its_original_layout(tmp_path):
    save_file({"weight": torch.arange(24).reshape(4, 6).float()}, tmp_path / "model.safetensors")
    mp.spawn(_check_sharded_meta_load, args=(str(tmp_path),), nprocs=2, join=True)


@pytest.mark.parametrize(
    "load_after_shard,loader",
    [(None, "missing"), (False, "missing"), (True, "missing"), (True, "noncallable"), (True, "supported")],
)
def test_after_shard_declaration_validates_loader_before_model_patches(load_after_shard, loader):
    from unittest.mock import Mock

    def build_model(*args, **kwargs):
        module = torch.nn.Linear(2, 2)
        if loader == "supported":
            module.load_hf_checkpoint_shards = Mock()
        elif loader == "noncallable":
            module.load_hf_checkpoint_shards = None
        return module

    auto_class = SimpleNamespace(
        from_config=Mock(side_effect=build_model),
        from_pretrained=Mock(side_effect=build_model),
    )
    hf_config = SimpleNamespace(tie_word_embeddings=False)
    if load_after_shard is not None:
        hf_config.load_checkpoint_after_shard = load_after_shard
    engine = object.__new__(FSDPEngine)
    engine.device_mesh = None
    engine.engine_config = SimpleNamespace(model_dtype="fp32", forward_only=False)
    engine.model_config = SimpleNamespace(
        model_type="language_model",
        hf_config=hf_config,
        trust_remote_code=False,
        local_path="checkpoint",
        use_liger=False,
        fused_kernel_options={},
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
    )
    engine.use_remove_padding = False
    engine.ulysses_sequence_parallel_size = 1
    with (
        patch("verl.utils.model.get_hf_auto_model_class", return_value=auto_class),
        patch("verl.workers.engine.fsdp.transformer_impl.get_init_weight_context_manager", return_value=nullcontext),
        patch("verl.workers.engine.fsdp.transformer_impl.apply_monkey_patch") as patch_model,
    ):
        if load_after_shard and loader != "supported":
            with pytest.raises(ValueError, match="load_checkpoint_after_shard=True.*load_hf_checkpoint_shards"):
                engine._build_module()
            patch_model.assert_not_called()
        else:
            module = engine._build_module()
            assert isinstance(module, torch.nn.Linear)
            assert module.weight.is_meta == bool(load_after_shard)
            patch_model.assert_called_once()
    if load_after_shard:
        auto_class.from_config.assert_called_once()
        auto_class.from_pretrained.assert_not_called()
    else:
        auto_class.from_pretrained.assert_called_once()
        auto_class.from_config.assert_not_called()
