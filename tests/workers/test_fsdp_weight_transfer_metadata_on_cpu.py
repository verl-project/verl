# Copyright 2026 Individual Contributor
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
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


@pytest.mark.parametrize("model_type", ["qwen3_moe", "gpt_oss"])
def test_metadata_matches_full_export_order_shapes_and_dtypes(model_type, monkeypatch):
    weights = {
        "new.linear.weight": torch.ones(3, 4),
        "layer.mlp.experts.gate_up_proj": torch.ones(2, 8, 3, dtype=torch.bfloat16),
        "layer.mlp.experts.down_proj": torch.ones(2, 3, 4, dtype=torch.bfloat16),
        "scalar": torch.tensor(1, dtype=torch.int64),
    }
    engine = object.__new__(FSDPEngine)
    engine.module = SimpleNamespace(state_dict=lambda: weights, _checkpoint_conversion_mapping={"old": "new"})
    engine.engine_config = SimpleNamespace(strategy="fsdp2")
    engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type), lora={})
    engine._uses_fsdp2_cpu_offload_policy = True
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda _: 2)
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    engine._is_offload_param = False
    stream, _ = engine.get_per_tensor_param()
    expected = [(name, value.dtype, tuple(value.shape)) for name, value in stream]
    assert expected[0] == ("old.linear.weight", torch.float32, (3, 4))
    assert engine.get_weight_transfer_metadata() == expected

    # Describing a sharded parameter only needs its global shape and dtype.
    # An attempt to gather, move or read storage would fail on these descriptors.
    engine.module.state_dict = lambda: {
        key: SimpleNamespace(shape=value.shape, dtype=value.dtype) for key, value in weights.items()
    }
    assert engine.get_weight_transfer_metadata() == expected


def _export_on_sharded_rank(rank, rendezvous_file):
    dist.init_process_group(
        backend="gloo", init_method=f"file://{rendezvous_file}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        module = torch.nn.Linear(3, 4, bias=False)
        fully_shard(module, mesh=init_device_mesh("cpu", (2,)))
        engine = object.__new__(FSDPEngine)
        engine.module = module
        engine.engine_config = SimpleNamespace(strategy="fsdp2")
        engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="linear"), lora={})
        engine._uses_fsdp2_cpu_offload_policy = False
        engine._is_offload_param = False

        assert isinstance(module.weight, DTensor)
        local_weight = module.weight.to_local()
        assert local_weight.shape == (2, 3)
        with (
            torch.no_grad(),
            patch.object(transformer_impl, "get_device_id", return_value="cpu"),
            patch.object(transformer_impl, "log_gpu_memory_usage"),
        ):
            for step in range(2):
                local_weight.fill_(2 * step + rank)
                with patch.object(DTensor, "full_tensor", side_effect=AssertionError("metadata gathered weights")):
                    assert engine.get_weight_transfer_metadata() == [("weight", torch.float32, (4, 3))]
                weights, _ = engine.get_per_tensor_param()
                [(name, full_weight)] = list(weights)
                assert name == "weight"
                expected = torch.tensor([2 * step, 2 * step + 1], dtype=torch.float32)
                expected = expected.repeat_interleave(2).view(4, 1).expand(4, 3)
                torch.testing.assert_close(full_weight, expected)
    finally:
        dist.destroy_process_group()


def test_two_fsdp_ranks_export_both_versions(tmp_path):
    mp.spawn(_export_on_sharded_rank, args=(str(tmp_path / "fsdp_rendezvous"),), nprocs=2)
