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

"""Exercise the real model-row functions without installing device-only MCore."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional
from unittest.mock import Mock, patch

import pytest
import torch
from torch.nested._internal.nested_tensor import NestedTensor

ROOT = Path(__file__).resolve().parents[3]


def _functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)


_row_functions = {"torch": torch}
_functions("verl/utils/megatron/router_replay_utils.py", {"prepare_model_row_routes"}, _row_functions)
prepare_model_row_routes = _row_functions["prepare_model_row_routes"]


@pytest.fixture
def functions():
    namespace = dict(
        torch=torch,
        Optional=Optional,
        NestedTensor=NestedTensor,
        prepare_model_row_routes=prepare_model_row_routes,
    )
    _functions(
        "verl/utils/megatron/router_replay_utils.py",
        {"merge_router_topk_indices", "set_router_replay_data"},
        namespace,
    )
    # Load the complete replay module; only unused external MCore endpoints are
    # isolated. The replay class, imports and gate math execute unchanged.
    external = {
        f"megatron.core.transformer.{name}": Mock()
        for name in ("moe.moe_utils", "moe.token_dispatcher", "moe.router", "transformer_config")
    }
    spec = importlib.util.spec_from_file_location(
        "router_replay_cpu", ROOT / "verl/utils/megatron/router_replay_patch.py"
    )
    replay = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, external):
        spec.loader.exec_module(replay)
    namespace.update(vars(replay))
    return namespace


def test_r3_preserves_ids_and_sequence_major_order(functions):
    ids = torch.nested.as_nested_tensor(
        [torch.tensor([[[256, 895]], [[7, 8]]]), torch.tensor([[[9, 10]]])], layout=torch.jagged
    )
    dense_ids, mask = prepare_model_row_routes(ids, torch.ones(ids.shape[0], 4, dtype=torch.bool))
    assert dense_ids[0, :, 0].tolist() == [[256, 895], [9, 10], [7, 8], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    assert mask.tolist() == [[True, True, True, False, False, False, False, False]]


def test_model_rows_reject_runtime_overflow_and_dense_input():
    ids = torch.nested.as_nested_tensor([torch.zeros(3, 1, 2, dtype=torch.long)], layout=torch.jagged)
    with pytest.raises(ValueError, match="runtime sequence"):
        prepare_model_row_routes(ids, torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="jagged"):
        prepare_model_row_routes(torch.zeros(1, 3, 1, 2), torch.ones(1, 3, dtype=torch.bool))


def test_r2_record_and_replay_preserve_expanded_image_rows(functions):
    # Two dense model sequences flattened [sequence, batch], each row distinct.
    rows = torch.arange(8 * 2 * 2).reshape(16, 2)
    router = SimpleNamespace(recorded_topk_idx=rows)
    functions.update(
        iter_model_routers=lambda model, router=router: [(1, router)],
        device_name="cpu",
        gather_from_sequence_parallel_region=lambda tensor, **kwargs: tensor,
    )
    raw = torch.nested.as_nested_tensor([torch.tensor([10, 99, 11]), torch.tensor([20, 99])], layout=torch.jagged)
    recorded = []
    functions["merge_router_topk_indices"](
        None,
        raw,
        recorded,
        SimpleNamespace(sequence_parallel=True),
        model=object(),
        model_sequence_length=8,
        model_row_lengths=[6, 5],
    )
    assert recorded[0].offsets().tolist() == [0, 6, 11]
    dense, mask = prepare_model_row_routes(recorded[0], torch.ones(2, 8, dtype=torch.bool))
    expected = rows.reshape(8, 2, 2)
    torch.testing.assert_close(dense[0, :10, 0].long(), expected[:5].reshape(-1, 2))
    assert dense[0, 10, 0].tolist() == expected[5, 0].tolist()
    assert mask.tolist() == [[True] * 11 + [False] * 5]


def test_backward_replay_is_fifo_and_masks_select_native_routes(functions):
    replay = functions["RouterReplay"]()
    for ids, mask in (([[2, 0]], [True]), ([[0, 2]], [False]), ([[0, 1]], [True])):
        replay.set_target_indices(torch.tensor(ids), torch.tensor(mask))
    replay.set_router_replay_action(functions["RouterReplayAction"].REPLAY_BACKWARD)
    scores = torch.tensor([[0.1, 0.7, 0.2]], requires_grad=True)

    def native_topk(value, k, **kwargs):
        return value.topk(k)

    for expected_ids in ([[2, 0]], [[1, 2]], [[0, 1]], [[1, 2]]):
        probs, ids = replay.get_replay_topk(scores, 2, default_compute_topk=native_topk)
        assert ids.tolist() == expected_ids
        torch.testing.assert_close(probs, scores.gather(1, torch.tensor(expected_ids)))
    replay.clear_indices()
    replay.set_router_replay_action(functions["RouterReplayAction"].REPLAY_FORWARD)
    probs, ids = replay.get_replay_topk(scores, 2, default_compute_topk=native_topk)
    torch.testing.assert_close(ids, scores.topk(2).indices)
    torch.testing.assert_close(probs, scores.topk(2).values)


@pytest.mark.parametrize("score_function,use_pre_softmax", [("softmax", False), ("softmax", True), ("sigmoid", False)])
def test_replayed_ids_use_current_gate_and_change_after_optimizer_steps(functions, score_function, use_pre_softmax):
    replay = functions["RouterReplay"]()
    ids = torch.tensor([[0, 2]])
    replay.set_target_indices(ids, torch.tensor([True]))
    replay.set_router_replay_action(functions["RouterReplayAction"].REPLAY_FORWARD)
    logits = torch.nn.Parameter(torch.tensor([[0.2, 0.3, -0.4]]))
    optimizer = torch.optim.SGD([logits], lr=0.5)
    outputs = []
    for _ in range(3):
        optimizer.zero_grad()
        probs, route_map = functions["_patched_topk_routing_with_score_function"](
            logits, 2, use_pre_softmax, None, None, score_function, None, False, replay, 1.5
        )
        assert route_map.tolist() == [[True, False, True]]
        if score_function == "sigmoid":
            current = logits.sigmoid().gather(1, ids)
            expected = current / current.sum(-1, keepdim=True)
        elif use_pre_softmax:
            expected = logits.softmax(-1).gather(1, ids)
        else:
            expected = logits.gather(1, ids).softmax(-1)
        torch.testing.assert_close(probs.gather(1, ids), expected * 1.5)
        outputs.append(probs.detach().clone())
        loss = probs[0, 0]
        actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad((expected * 1.5)[0, 0], logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        logits.grad = actual_grad
        optimizer.step()
    assert not torch.allclose(outputs[0], outputs[-1])


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_replay_scatter_keeps_ids_and_mask_together(functions, sequence_parallel):
    ids = torch.nested.as_nested_tensor(
        [torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]]]), torch.tensor([[[6, 7]]])], layout=torch.jagged
    )
    full_ids, full_mask = prepare_model_row_routes(ids, torch.ones(ids.shape[0], 4, dtype=torch.bool))
    for rank in range(2):
        router = functions["RouterReplay"]()
        scatter = Mock(side_effect=lambda tensor, rank=rank: tensor.chunk(2)[rank])
        functions.update(
            iter_model_routers=lambda model, router=router: [(1, router)],
            device_name="cpu",
            get_thd_quant_align_size=lambda config: 1,
            _context_parallel_layout=lambda config: "contiguous",
            scatter_to_sequence_parallel_region=scatter,
        )
        functions["set_router_replay_data"](
            full_ids,
            None,
            SimpleNamespace(seq_length=8, sequence_parallel=sequence_parallel, num_layers=1),
            model=object(),
            preprocessed=True,
            replay_mask=full_mask,
        )
        rows = slice(rank * 4, (rank + 1) * 4) if sequence_parallel else slice(None)
        torch.testing.assert_close(router.target_topk_idx, full_ids[0, rows, 0])
        torch.testing.assert_close(router.target_replay_mask, full_mask[0, rows])
        assert scatter.call_count == 2 * int(sequence_parallel)


def test_actual_engine_replay_rejects_missing_routes(functions):
    tree = ast.parse((ROOT / "verl/workers/engine/megatron/transformer_impl.py").read_text())
    engine_class = next(node for node in tree.body if getattr(node, "name", None) == "MegatronEngineWithLMHead")
    method = next(node for node in engine_class.body if getattr(node, "name", None) == "forward_step")
    namespace = dict(
        functions,
        Iterator=Iterator,
        TensorDict=object,
        unwrap_model=lambda model: model,
        get_device_id=lambda: "cpu",
        DatasetPadMode=SimpleNamespace(NO_PADDING=0),
        RouterReplayHelper=SimpleNamespace(
            is_replay_backward_action=lambda *args, **kwargs: False,
            is_replay_forward_action=lambda *args, **kwargs: True,
        ),
        tu=SimpleNamespace(get_non_tensor_data=lambda batch=None, key=None, default=None, **kwargs: default),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_engine_forward_step", "exec"), namespace)

    class Batch(dict):
        def to(self, device):
            return self

    batch = Batch(temperature=1.0)
    config = SimpleNamespace(
        use_fused_kernels=False,
        pad_to_length=False,
        dynamic_context_parallel=False,
        router_replay=SimpleNamespace(mode="R3"),
    )
    engine = SimpleNamespace(
        engine_config=config,
        model_config=SimpleNamespace(hf_config=object()),
        enable_routing_replay=True,
        tf_config=object(),
        _get_context_parallel_layout=lambda model: "contiguous",
        prepare_model_inputs=lambda batch: dict(
            input_ids=object(),
            attention_mask=None,
            multi_modal_inputs={},
            loss_mask=None,
            routed_experts=None,
        ),
    )
    with pytest.raises(RuntimeError, match="requires routes"):
        namespace["forward_step"](engine, iter([batch]), object(), None, None)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("post_process", [False, True])
def test_generic_forwards_preserve_r2_record_and_r3_replay(functions, monkeypatch, fused, record, post_process):
    import sys
    import types

    _functions(
        "verl/utils/megatron/router_replay_utils.py",
        {"align_r3_router_replay_data", "build_r3_replay_mask"},
        functions,
    )
    replay, merge = Mock(), Mock()
    module = types.ModuleType("verl.utils.megatron.router_replay_utils")
    module.align_r3_router_replay_data = functions["align_r3_router_replay_data"]
    module.build_r3_replay_mask = functions["build_r3_replay_mask"]
    module.set_router_replay_data = replay
    module.merge_router_topk_indices = merge
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = dict(
        functions,
        Tensor=torch.Tensor,
        unwrap_model=lambda model: model,
        get_thd_quant_align_size=lambda config: 1,
        preprocess_thd_engine=lambda value, **kwargs: (value.values().reshape(1, -1), None, None),
        postprocess_thd_engine=lambda value, *args, **kwargs: value,
        _use_output_processor_hook=lambda model: False,
        get_fused_impl_backend=lambda model: "triton",
    )
    if fused:
        _functions("verl/models/mcore/model_forward_fused.py", {"fused_forward_model_engine"}, namespace)
        forward = namespace["fused_forward_model_engine"]()
    else:
        _functions("verl/models/mcore/model_forward.py", {"gptmodel_forward_model_engine"}, namespace)
        forward = namespace["gptmodel_forward_model_engine"]

    class Model:
        pre_process = True
        config = SimpleNamespace()
        vp_stage = 1

        def __call__(self, **kwargs):
            if not record:
                assert replay.call_count == 1
            return SimpleNamespace(log_probs=torch.zeros(3), entropy=torch.zeros(3)) if fused else torch.zeros(1, 3)

    model = Model()
    model.post_process = post_process
    input_ids = torch.nested.as_nested_tensor([torch.tensor([4, 5, 6])], layout=torch.jagged)
    routes = torch.nested.as_nested_tensor([torch.tensor([[[0, 1]], [[1, 2]]])], layout=torch.jagged)
    sink = []
    kwargs = (
        dict(recorded_routes=sink) if record else dict(routed_experts=routes, replay_response_mask=torch.tensor([[1]]))
    )
    if fused:
        kwargs.update(labels=input_ids, temperature=1.0, calculate_entropy=False, pad_token_id=0)
    forward(model, input_ids=input_ids, multi_modal_inputs={}, **kwargs)
    assert replay.call_count == int(not record)
    assert merge.call_count == int(record)
    if record:
        assert merge.call_args.args[2] is sink
        assert merge.call_args.kwargs["vp_rank"] == 1
    else:
        aligned = replay.call_args.args[0]
        assert aligned.offsets().tolist() == [0, 3]
        torch.testing.assert_close(aligned.values()[:2], routes.values())
        assert replay.call_args.kwargs["replay_mask"].values().tolist() == [True, True, False]
        assert replay.call_args.kwargs["vp_rank"] == 1
