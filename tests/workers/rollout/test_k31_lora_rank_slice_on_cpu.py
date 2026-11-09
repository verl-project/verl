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
"""Per-rank slicing of an expert LoRA adapter for the colocated rollout sync (VERL_K31_LORA_RANK_SLICE=1)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import verl

# Load the module by path: importing the vllm_rollout package needs vLLM, the module itself does not.
_PATH = Path(verl.__file__).parent / "workers/rollout/vllm_rollout/k31_lora_rank_slice.py"
_SPEC = importlib.util.spec_from_file_location("k31_lora_rank_slice", _PATH)
rs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rs)

W1_B = "model.layers.3.mlp.experts.w1.lora_B.weight"
W2_A = "model.layers.3.mlp.experts.w2.lora_A.weight"
W1_A = "model.layers.3.mlp.experts.w1.lora_A.weight"
DENSE = "model.layers.3.self_attn.q_b_proj.lora_A.weight"


def _adapter(experts=8, inter=16, rank=4, hidden=12):
    return [
        (W1_B, torch.arange(experts * inter * rank, dtype=torch.float32).reshape(experts, inter, rank)),
        (W2_A, torch.arange(experts * rank * inter, dtype=torch.float32).reshape(experts, rank, inter)),
        (W1_A, torch.ones(1, rank, hidden)),
        (DENSE, torch.ones(rank, hidden)),
    ]


def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv(rs.ENV, raising=False)
    assert not rs.enabled()
    monkeypatch.setenv(rs.ENV, "1")
    assert rs.enabled()


@pytest.mark.parametrize("tp_rank", [0, 1, 3])
def test_tp_slice_adapter_narrows_only_tp_sharded_expert_factors(tp_rank):
    full = dict(_adapter())
    out = list(rs.tp_slice_adapter(_adapter(), tp_rank, 4))
    assert out[0][0] == rs.TP_TAG and out[0][1].tolist() == [tp_rank, 4]
    got = dict(out[1:])
    assert torch.equal(got[W1_B], full[W1_B][:, 4 * tp_rank : 4 * (tp_rank + 1), :])
    assert torch.equal(got[W2_A], full[W2_A][:, :, 4 * tp_rank : 4 * (tp_rank + 1)])
    assert got[W1_B].is_contiguous() and got[W2_A].is_contiguous()
    assert got[W1_A] is full[W1_A] or torch.equal(got[W1_A], full[W1_A])
    assert torch.equal(got[DENSE], full[DENSE])


def test_tp_slice_adapter_rejects_bad_layouts():
    with pytest.raises(ValueError):
        list(rs.tp_slice_adapter(_adapter(), 4, 4))
    with pytest.raises(ValueError):
        list(rs.tp_slice_adapter(_adapter(inter=15), 0, 4))


def _ep_spec(ep_rank, ep_size, experts):
    return SimpleNamespace(ep_rank=ep_rank, global_num_experts=experts, local_num_experts=experts // ep_size)


def test_localize_ep_strips_tag_and_checks_placement():
    local = {
        rs.EP_TAG: torch.tensor([2, 4, 8]),
        W1_B: torch.zeros(2, 16, 4),
        W1_A: torch.zeros(1, 4, 12),
        DENSE: torch.zeros(4, 12),
    }
    out = rs.localize_ep(dict(local), _ep_spec(2, 4, 8))
    assert rs.EP_TAG not in out and set(out) == {W1_B, W1_A, DENSE}

    with pytest.raises(RuntimeError, match="placement mismatch"):
        rs.localize_ep(dict(local), _ep_spec(1, 4, 8))
    with pytest.raises(RuntimeError, match="without expert parallelism"):
        rs.localize_ep(dict(local), None)
    with pytest.raises(RuntimeError, match="linear expert placement"):
        rs.localize_ep(dict(local), _ep_spec(2, 4, 8), placement="round_robin")
    bad = dict(local, **{W1_B: torch.zeros(8, 16, 4)})
    with pytest.raises(RuntimeError, match="expert dim 8"):
        rs.localize_ep(bad, _ep_spec(2, 4, 8))


def test_localize_tp_checks_shard_against_every_moe_layer(monkeypatch):
    monkeypatch.setattr(rs, "install_presliced_tp_passthrough", lambda: True)
    moe = SimpleNamespace(tp_rank=1, tp_size=4, _tp_shard_size=4, use_ep=False)
    sliced = dict(rs.tp_slice_adapter(_adapter(), 1, 4))
    out = rs.localize_tp(dict(sliced), [moe, moe])
    assert rs.TP_TAG not in out and out[W1_B].shape == (8, 4, 4)

    with pytest.raises(RuntimeError, match="layout mismatch"):
        rs.localize_tp(dict(sliced), [SimpleNamespace(tp_rank=0, tp_size=4, _tp_shard_size=4, use_ep=False)])
    whole = dict(_adapter(), **{rs.TP_TAG: torch.tensor([1, 4])})
    with pytest.raises(RuntimeError, match="expert shard 4"):
        rs.localize_tp(whole, [moe])


def test_localize_received_passes_untagged_adapter_through():
    weights = dict(_adapter())
    assert rs.localize_received(weights, worker=None) is weights
