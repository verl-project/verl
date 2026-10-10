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

"""Exercise the production exporter without importing CUDA-only VeOmni deps."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


def exporter(namespace):
    source = Path(__file__).parents[3] / "verl/workers/engine/veomni/transformer_impl.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VeOmniEngine")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_per_tensor_param"
    )
    code = ast.Module(body=[method], type_ignores=[])
    namespace.update(
        torch=torch,
        DTensor=DTensor,
        convert_weight_keys=lambda params, module: params,
        get_device_id=lambda: "cpu",
        parallel_state=SimpleNamespace(get_parallel_state=lambda: SimpleNamespace(ep_enabled=False)),
        get_moe_param_handler=lambda *args: None,
    )
    exec(compile(code, str(source), "exec"), namespace)
    return namespace["get_per_tensor_param"]


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("cpu_policy", [False, True])
@pytest.mark.parametrize("converter_kind", ["absent", "none", "load_only"])
def test_cpu_sharded_export_preserves_values_without_whole_model_staging(tmp_path, offload, cpu_policy, converter_kind):
    owns_process_group = not torch.distributed.is_initialized()
    if owns_process_group:
        torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    else:
        # The checkpoint tests can own a single-rank Gloo group for the session.
        assert torch.distributed.get_backend() == "gloo"
        assert torch.distributed.get_world_size() == 1
    try:
        mesh = torch.distributed.init_device_mesh("cpu", (1,))
        expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        module = torch.nn.Module()
        module.config = SimpleNamespace(model_type="qwen3_moe")
        module.register_parameter("weight", torch.nn.Parameter(expected.clone()))
        module.register_buffer("buffer", torch.tensor([7.0]))
        fully_shard(module, mesh=mesh)
        module.cpu()
        converter = SimpleNamespace(can_handle=lambda name: False, convert=lambda name, tensor: tensor)
        if converter_kind != "absent":
            module._create_checkpoint_tensor_converter = lambda model: (
                converter if converter_kind == "load_only" else None
            )

        def get_converter(model):
            factory = getattr(model, "_create_checkpoint_tensor_converter", None)
            return factory(model) if callable(factory) else None

        forbidden = Mock(side_effect=AssertionError("whole-model staging is unnecessary"))
        method = exporter(
            dict(
                load_veomni_model_to_gpu=forbidden,
                offload_veomni_model_to_cpu=forbidden,
                get_checkpoint_tensor_converter=get_converter,
            )
        )
        engine = SimpleNamespace(module=module, _is_offload_param=offload, _uses_fsdp2_cpu_offload_policy=cpu_policy)
        params, peft = method(engine)
        output = dict(params)
        assert peft is None
        torch.testing.assert_close(output["weight"], expected)
        torch.testing.assert_close(output["buffer"], torch.tensor([7.0]))
        assert isinstance(module.weight, DTensor)
        assert module.weight.device.type == "cpu"
        forbidden.assert_not_called()
    finally:
        if owns_process_group:
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize("cpu_policy", [False, True])
def test_custom_converter_keeps_existing_placement(cpu_policy):
    events = []
    module = SimpleNamespace(
        _create_checkpoint_tensor_converter=lambda: None,
        state_dict=Mock(side_effect=AssertionError("export-capable converter must bypass state_dict")),
    )
    converter = SimpleNamespace(export_weights=lambda module: events.append("export") or iter(()))
    method = exporter(
        dict(
            load_veomni_model_to_gpu=lambda module: events.append("load"),
            get_checkpoint_tensor_converter=lambda module: converter,
        )
    )
    method(SimpleNamespace(module=module, _uses_fsdp2_cpu_offload_policy=cpu_policy))
    assert events == (["export"] if cpu_policy else ["load", "export"])
    module.state_dict.assert_not_called()
