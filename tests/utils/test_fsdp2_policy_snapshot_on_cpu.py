# Copyright 2026 Individual Contributor: Tiantian Qiao
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

import copy
import datetime
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_sharded_load_from_cpu, fsdp2_sharded_save_to_cpu
from verl.workers.config import FSDPEngineConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, hidden):
        return torch.tanh(self.proj(hidden))


class _Model(torch.nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self, tied):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        self.embed_tokens = torch.nn.Embedding(16, 8)
        self.layers = torch.nn.ModuleList([_Block(), _Block()])
        self.lm_head = torch.nn.Linear(8, 16, bias=False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, ids):
        hidden = self.embed_tokens(ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(hidden)


def _reference_output(model, ids, mixed_precision):
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed_precision, cache_enabled=False):
        return model(ids)


def _check_restore(
    rank,
    rendezvous,
    tied,
    reshard_after_forward,
    configured_size,
    world_size=2,
    save_after_eval=False,
    mixed_precision=False,
):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        torch.manual_seed(123)
        model = _Model(tied)
        reference = copy.deepcopy(model)
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            old_output = _reference_output(reference, ids, mixed_precision)
            for parameter in reference.parameters():
                parameter.add_(0.25)
            current_output = _reference_output(reference, ids, mixed_precision)

        mesh = init_device_mesh("cpu", (world_size,))
        policy = (
            MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            if mixed_precision
            else MixedPrecisionPolicy()
        )
        apply_fsdp2(model, {"mesh": mesh, "reshard_after_forward": reshard_after_forward, "mp_policy": policy}, {})
        # Keep engine initialization small while using its real eval context and FSDP wrappers.
        engine = object.__new__(FSDPEngineWithLMHead)
        engine.module = model
        engine.engine_config = FSDPEngineConfig(
            strategy="fsdp2", fsdp_size=configured_size, reshard_after_forward=reshard_after_forward
        )
        engine._is_offload_param = False
        engine._is_offload_optimizer = False
        engine.ulysses_parallel_group = None
        engine.mode = None

        old, old_spec = fsdp2_sharded_save_to_cpu(model)
        with torch.no_grad():
            for parameter in model.parameters():
                assert isinstance(parameter, DTensor)
                parameter.to_local().add_(0.25)
        current, current_spec = fsdp2_sharded_save_to_cpu(model)

        fsdp2_sharded_load_from_cpu(model, old, old_spec)
        with torch.no_grad(), engine.eval_mode():
            torch.testing.assert_close(model(ids), old_output)

        assert engine.mode is None
        if save_after_eval:
            after_eval, _ = fsdp2_sharded_save_to_cpu(model)
            for name in old:
                torch.testing.assert_close(after_eval[name][0], old[name][0], rtol=0, atol=0)
        fsdp2_sharded_load_from_cpu(model, current, current_spec)
        # Saving here would reshard the model and could discard a stale full-parameter cache.
        for name, param in model.named_parameters():
            assert isinstance(param, DTensor)
            torch.testing.assert_close(param.to_local(), current[name][0], rtol=0, atol=0)
        with torch.no_grad(), engine.eval_mode():
            torch.testing.assert_close(model(ids), current_output)
        after_current_eval, _ = fsdp2_sharded_save_to_cpu(model)
        for name in current:
            torch.testing.assert_close(after_current_eval[name][0], current[name][0], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("reshard_after_forward", [True, False], ids=["reshard", "keep-full"])
@pytest.mark.parametrize("configured_size", [-1, 2], ids=["auto-size", "explicit-size"])
def test_policy_switch_after_eval(tmp_path, tied, reshard_after_forward, configured_size):
    mp.spawn(
        _check_restore,
        args=(str(tmp_path / "rendezvous"), tied, reshard_after_forward, configured_size),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("reshard_after_forward", [True, False], ids=["reshard", "keep-full"])
def test_single_rank_policy_switch(tmp_path, reshard_after_forward):
    mp.spawn(
        _check_restore,
        args=(str(tmp_path / "rendezvous"), True, reshard_after_forward, 1, 1),
        nprocs=1,
        join=True,
    )


def test_save_after_eval_uses_local_shards(tmp_path):
    mp.spawn(
        _check_restore,
        args=(str(tmp_path / "rendezvous"), True, False, -1, 2, True),
        nprocs=2,
        join=True,
    )


def test_mixed_precision_policy_switch(tmp_path):
    mp.spawn(
        _check_restore,
        args=(str(tmp_path / "rendezvous"), True, False, -1, 2, False, True),
        nprocs=2,
        join=True,
    )
