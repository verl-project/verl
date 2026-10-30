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
"""THD inputs for VLMs that shard the full packed stream themselves (GLM-5.3-Flash)."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

import verl.utils.device as device_module


def _load_mcore_util(monkeypatch, tp_size: int = 1, cp_size: int = 1, cp_rank: int = 0):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")

    parallel_state.get_context_parallel_world_size = lambda: cp_size
    parallel_state.get_context_parallel_rank = lambda: cp_rank
    parallel_state.get_context_parallel_group = lambda: object()
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size
    parallel_state.get_dynamic_data_context_parallel_groups = lambda group_size: ("dcp", group_size)

    class PackedSeqParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    packed_seq_params.PackedSeqParams = PackedSeqParams
    core.parallel_state = parallel_state
    megatron.core = core
    megatron_utils = types.ModuleType("verl.utils.megatron_utils")
    megatron_utils.unwrap_model = lambda model: model
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed_seq_params)
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_utils", megatron_utils)
    monkeypatch.setattr(device_module, "is_npu_available", False)

    util_path = Path(__file__).parents[2] / "verl" / "models" / "mcore" / "util.py"
    spec = importlib.util.spec_from_file_location("mcore_util_full_stream", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested(*rows):
    return torch.nested.nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)


@pytest.mark.parametrize("need_roll", [False, True])
def test_full_stream_matches_the_unsharded_engine_layout(monkeypatch, need_roll):
    util = _load_mcore_util(monkeypatch, tp_size=4)
    values = _nested([1, 2, 3], [4, 5, 6, 7, 8])
    expected, packed_seq_params, _ = util.preprocess_thd_engine(values, pre_process=True, need_roll=need_roll)

    stream = util.preprocess_thd_full_stream(values, packed_seq_params, need_roll=need_roll)

    assert packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 4, 12]
    torch.testing.assert_close(stream, expected)


def test_full_stream_keeps_every_cp_rank_shard(monkeypatch):
    """With CP, the full stream is the padded layout that each rank's zigzag shard is cut from."""
    util = _load_mcore_util(monkeypatch, cp_size=2, cp_rank=1)
    values = _nested([1, 2, 3], [4, 5, 6, 7, 8])
    local, packed_seq_params, _ = util.preprocess_thd_engine(values, pre_process=True)

    stream = util.preprocess_thd_full_stream(values, packed_seq_params)

    torch.testing.assert_close(stream, torch.tensor([[1, 2, 3, 0, 4, 5, 6, 7, 8, 0, 0, 0]]))
    # Rank 1 holds the second and third quarter-chunks of each sequence.
    torch.testing.assert_close(local, stream[:, [1, 2, 6, 7, 8, 9]])


@pytest.mark.parametrize(
    ("cp_size", "thd_kwargs", "message"),
    [
        (1, {"local_cp_size": 2}, "dynamic context parallelism"),
        (2, {"cp_layout": "contiguous"}, "zigzag"),
    ],
)
def test_full_stream_rejects_cp_layouts_the_model_cannot_reproduce(monkeypatch, cp_size, thd_kwargs, message):
    util = _load_mcore_util(monkeypatch, cp_size=cp_size)
    with pytest.raises(ValueError, match=message):
        util.check_full_stream_cp_layout(**thd_kwargs)
    util.check_full_stream_cp_layout(cp_layout="contiguous" if cp_size == 1 else "zigzag", min_local_rows=None)


def test_vlm_thd_inputs_route_full_stream_models(monkeypatch):
    util = _load_mcore_util(monkeypatch)
    monkeypatch.setattr(util, "takes_full_packed_thd_vlm_inputs", lambda model: True)
    values = _nested([1, 2, 3], [4, 5])
    input_ids_rmpad, packed_seq_params, _ = util.preprocess_thd_engine(values, pre_process=True)

    input_ids, attention_mask, position_ids = util.preprocess_vlm_thd_engine(
        object(), values, input_ids_rmpad, packed_seq_params, None, 0, cp_layout="zigzag", local_cp_size=None
    )

    torch.testing.assert_close(input_ids, torch.tensor([[1, 2, 3, 4, 5]]))
    assert attention_mask is None and position_ids is None


class _LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return "output"


def test_language_model_forward_kwargs_injects_and_restores(monkeypatch):
    util = _load_mcore_util(monkeypatch)
    wrapper = torch.nn.Module()
    wrapper.language_model = _LanguageModel()

    with util.language_model_forward_kwargs(wrapper, compute_mtp_loss=False, output_processor="hook"):
        assert wrapper.language_model.forward(input_ids=1) == "output"

    assert wrapper.language_model.calls == [{"input_ids": 1, "compute_mtp_loss": False, "output_processor": "hook"}]
    assert "forward" not in vars(wrapper.language_model)
    wrapper.language_model.forward(input_ids=2)
    assert wrapper.language_model.calls[-1] == {"input_ids": 2}


def test_language_model_forward_kwargs_restores_an_existing_patch(monkeypatch):
    util = _load_mcore_util(monkeypatch)
    wrapper = torch.nn.Module()
    wrapper.language_model = _LanguageModel()
    patched = wrapper.language_model.forward
    wrapper.language_model.forward = patched

    with pytest.raises(RuntimeError):
        with util.language_model_forward_kwargs(wrapper, compute_mtp_loss=True):
            raise RuntimeError

    assert vars(wrapper.language_model)["forward"] is patched
