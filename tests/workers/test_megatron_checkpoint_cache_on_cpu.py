# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_method(path: Path, class_name: str, method_name: str):
    module = ast.parse(path.read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    fn_node = deepcopy(item)
                    fn_node.decorator_list = []
                    fn_node.returns = None
                    for arg in fn_node.args.posonlyargs + fn_node.args.args + fn_node.args.kwonlyargs:
                        arg.annotation = None
                    if fn_node.args.vararg is not None:
                        fn_node.args.vararg.annotation = None
                    if fn_node.args.kwarg is not None:
                        fn_node.args.kwarg.annotation = None
                    fn_module = ast.Module(body=[fn_node], type_ignores=[])
                    ast.fix_missing_locations(fn_module)
                    namespace = {}
                    exec(compile(fn_module, filename=str(path), mode="exec"), namespace)
                    return namespace[method_name]
    raise AssertionError(f"Method {class_name}.{method_name} not found in {path}")


def _build_torch(events: list[str]):
    return SimpleNamespace(
        distributed=SimpleNamespace(
            barrier=lambda: events.append("barrier"),
        )
    )


def test_actor_checkpoint_save_clears_cache_after_save():
    events: list[str] = []
    path = REPO_ROOT / "verl" / "workers" / "megatron_workers.py"
    save_checkpoint = _load_method(path, "ActorRolloutRefWorker", "save_checkpoint")

    globals_dict = {
        "torch": _build_torch(events),
        "load_megatron_model_to_gpu": lambda module: events.append(f"load_model:{module}"),
        "load_megatron_optimizer": lambda optimizer: events.append(f"load_optim:{optimizer}"),
        "offload_megatron_model_to_cpu": lambda module: events.append(f"offload_model:{module}"),
        "offload_megatron_optimizer": lambda optimizer: events.append(f"offload_optim:{optimizer}"),
        "aggressive_empty_cache": lambda force_sync=True: events.append(f"empty_cache:{force_sync}"),
    }
    save_checkpoint.__globals__.update(globals_dict)

    checkpoint_manager = SimpleNamespace(
        checkpoint_config=SimpleNamespace(async_save=True),
        save_checkpoint=lambda **kwargs: events.append(f"save:{kwargs['global_step']}"),
    )
    worker = SimpleNamespace(
        _is_offload_param=True,
        _is_offload_optimizer=True,
        actor_module="actor_module",
        actor_optimizer="actor_optimizer",
        checkpoint_mananager=checkpoint_manager,
    )

    save_checkpoint(worker, "/tmp/ckpt", global_step=7)

    assert events == [
        "load_model:actor_module",
        "load_optim:actor_optimizer",
        "save:7",
        "barrier",
        "offload_model:actor_module",
        "offload_optim:actor_optimizer",
        "empty_cache:True",
    ]


def test_critic_checkpoint_save_clears_cache_after_save():
    events: list[str] = []
    path = REPO_ROOT / "verl" / "workers" / "megatron_workers.py"
    save_checkpoint = _load_method(path, "CriticWorker", "save_checkpoint")

    save_checkpoint.__globals__.update(
        {
            "torch": _build_torch(events),
            "load_megatron_model_to_gpu": lambda module: events.append(f"load_model:{module}"),
            "offload_megatron_model_to_cpu": lambda module: events.append(f"offload_model:{module}"),
            "aggressive_empty_cache": lambda force_sync=True: events.append(f"empty_cache:{force_sync}"),
        }
    )

    checkpoint_manager = SimpleNamespace(
        save_checkpoint=lambda **kwargs: events.append(f"save:{kwargs['global_step']}"),
    )
    worker = SimpleNamespace(
        _is_offload_param=True,
        critic_module="critic_module",
        checkpoint_mananager=checkpoint_manager,
    )

    save_checkpoint(worker, "/tmp/ckpt", global_steps=9)

    assert events == [
        "load_model:critic_module",
        "save:9",
        "barrier",
        "offload_model:critic_module",
        "empty_cache:True",
    ]


def test_engine_checkpoint_save_clears_cache_after_save():
    events: list[str] = []
    path = REPO_ROOT / "verl" / "workers" / "engine" / "megatron" / "transformer_impl.py"
    save_checkpoint = _load_method(path, "MegatronEngine", "save_checkpoint")

    save_checkpoint.__globals__.update(
        {
            "torch": _build_torch(events),
            "get_megatron_module_device": lambda module: "cuda",
            "load_megatron_model_to_gpu": lambda module, load_grad=True: events.append(
                f"load_model:{module}:{load_grad}"
            ),
            "offload_megatron_model_to_cpu": lambda module: events.append(f"offload_model:{module}"),
            "aggressive_empty_cache": lambda force_sync=True: events.append(f"empty_cache:{force_sync}"),
        }
    )

    checkpoint_manager = SimpleNamespace(
        save_checkpoint=lambda **kwargs: events.append(f"save:{kwargs['global_step']}"),
    )
    engine = SimpleNamespace(
        _is_offload_param=True,
        module="engine_module",
        checkpoint_mananager=checkpoint_manager,
    )

    save_checkpoint(engine, "/tmp/ckpt", global_step=11)

    assert events == [
        "load_model:engine_module:True",
        "save:11",
        "barrier",
        "offload_model:engine_module",
        "empty_cache:True",
    ]
