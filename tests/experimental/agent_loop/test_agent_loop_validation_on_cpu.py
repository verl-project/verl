# Copyright 2026 the verl contributors
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

"""Exercise native constructors and dispatch with real Hydra, without Ray/GPU imports."""

from __future__ import annotations

import ast
import asyncio
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_LOOP_SOURCE = REPO_ROOT / "verl/experimental/agent_loop/agent_loop.py"


def _load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name in names:
                definitions.append(node)
            else:
                definitions.extend(child for child in node.body if f"{node.name}.{getattr(child, 'name', '')}" in names)
        elif isinstance(node, ast.AsyncFunctionDef) and node.name in names:
            definitions.append(node)
    assert len(definitions) == len(names)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)


class _ContextAgent:
    """Hydra target populated with the production base constructor by the fixture."""


@pytest.fixture
def dispatch(monkeypatch):
    calls, postprocessed, statuses = [], [], []
    expected_output = object()

    async def run(self, sampling_params, **kwargs):
        await asyncio.sleep(0)
        calls.append((sampling_params, kwargs, self.rollout_context))
        return expected_output

    async def put_status(**kwargs):
        statuses.append(kwargs)

    registry = {"custom": {"_target_": f"{__name__}._ContextAgent"}}
    namespace = {
        "__name__": __name__,
        "asyncio": asyncio,
        "BaseModel": BaseModel,
        "ConfigDict": ConfigDict,
        "logger": logging.getLogger(__name__),
        "hydra": hydra,
        "rollout_trace_attr": lambda **kwargs: nullcontext(),
        "_agent_loop_registry": registry,
        "DictConfigWrap": lambda config: SimpleNamespace(config=config),
        "ToolListWrap": lambda tools: SimpleNamespace(tools=tools),
        "create_continuous_token_builder": lambda *args, **kwargs: object(),
        "get_event_loop": lambda: None,
        "tq": SimpleNamespace(async_kv_put=put_status),
    }
    _load_definitions(
        AGENT_LOOP_SOURCE,
        {"RolloutContext", "AgentLoopBase.__init__", "AgentLoopWorker._run_agent_loop"},
        namespace,
    )
    _load_definitions(
        REPO_ROOT / "verl/trainer/ppo/v1/agent_loop_tq.py",
        {"AgentLoopWorkerTQ._run_prompt", "_settle_session_tasks"},
        namespace,
    )
    monkeypatch.setattr(_ContextAgent, "__init__", namespace["__init__"])
    monkeypatch.setattr(_ContextAgent, "run", run, raising=False)

    class Worker:
        _run_agent_loop = namespace["_run_agent_loop"]
        _run_prompt = namespace["_run_prompt"]
        config = SimpleNamespace(
            data={},
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2, val_kwargs=SimpleNamespace(n=1))),
        )
        llm_client = object()
        teacher_client = {}
        tokenizer = object()
        processor = None
        hf_model_type = None
        dataset_cls = object()
        tools = []

        async def _agent_loop_postprocess(self, output, validate, **kwargs):
            assert output is expected_output
            postprocessed.append((validate, kwargs))
            return output

    return SimpleNamespace(
        worker=Worker(),
        registry=registry,
        context_type=namespace["RolloutContext"],
        calls=calls,
        postprocessed=postprocessed,
        statuses=statuses,
        output=expected_output,
    )


def _trajectory(validate, step):
    return {"validate": validate, "step": step, "sample_index": 0, "rollout_n": 0}


@pytest.mark.parametrize(("validate", "step"), [(False, 1), (True, 0), (True, 5)])
@pytest.mark.parametrize("conversion", ["none", "partial", "object", "all"])
@pytest.mark.parametrize("recursive", [False, True])
def test_constructor_receives_context_without_modifying_dataset_fields(dispatch, validate, step, conversion, recursive):
    dispatch.registry["custom"].update(
        {
            "_convert_": conversion,
            "_recursive_": recursive,
            "rollout_context": {"is_validation": not validate, "step": 999},
        }
    )
    sampling_params = {"temperature": 0.7}
    fields = {
        "raw_prompt": ["question"],
        "global_steps": step,
        "custom_field": "preserved",
        "rollout_context": {"source": "dataset"},
        "is_validation": "dataset-value",
        "step": "dataset-step",
    }
    original_fields = dict(fields)
    trajectory = _trajectory(validate, step)
    result = asyncio.run(
        dispatch.worker._run_agent_loop(sampling_params, trajectory, agent_name="custom", trace=False, **fields)
    )

    assert result is dispatch.output
    assert fields == original_fields
    assert len(dispatch.calls) == 1
    received_sampling, received_fields, context = dispatch.calls[0]
    assert received_sampling is sampling_params
    assert received_fields == original_fields
    assert "validate" not in received_fields
    assert isinstance(context, dispatch.context_type)
    assert context.is_validation is validate
    assert context.step == step
    trajectory.update(validate=not validate, step=999)
    assert context.is_validation is validate
    assert context.step == step
    assert dispatch.postprocessed == [(validate, original_fields)]
    assert sampling_params == {"temperature": 0.7}


def test_context_is_immutable(dispatch):
    context = dispatch.context_type(is_validation=True, step=0)
    with pytest.raises(ValidationError, match="frozen"):
        context.is_validation = False
    with pytest.raises(ValidationError, match="frozen"):
        context.step = 1


@pytest.mark.parametrize("supply_context", [False, True])
def test_direct_base_construction_does_not_infer_runtime_mode(dispatch, supply_context):
    worker = dispatch.worker
    context = dispatch.context_type(is_validation=True, step=0) if supply_context else None
    agent = _ContextAgent(
        trainer_config=SimpleNamespace(config=worker.config),
        server_manager=worker.llm_client,
        tokenizer=worker.tokenizer,
        processor=None,
        dataset_cls=worker.dataset_cls,
        data_config=SimpleNamespace(config=worker.config.data),
        **({"rollout_context": context} if supply_context else {}),
    )
    assert agent.rollout_context is context


def test_run_does_not_require_new_keyword_parameters(dispatch, monkeypatch):
    async def run(self, sampling_params, *, raw_prompt):
        assert raw_prompt == ["question"]
        assert self.rollout_context.is_validation is True
        assert self.rollout_context.step == 0
        return dispatch.output

    monkeypatch.setattr(_ContextAgent, "run", run)
    assert (
        asyncio.run(
            dispatch.worker._run_agent_loop(
                {}, _trajectory(True, 0), agent_name="custom", raw_prompt=["question"], trace=False
            )
        )
        is dispatch.output
    )


def test_context_is_exported_from_agent_loop_package():
    tree = ast.parse((AGENT_LOOP_SOURCE.parent / "__init__.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "agent_loop"
        for alias in node.names
    }
    exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    assert "RolloutContext" in imports
    assert "RolloutContext" in exports


def test_transfer_queue_dispatch_keeps_concurrent_training_and_validation_separate(dispatch):
    async def run_batches():
        await asyncio.gather(
            *(
                dispatch.worker._run_prompt(
                    {
                        "agent_name": "custom",
                        "uid": f"{step}-{index}",
                        "global_steps": step,
                        "raw_prompt": ["question"],
                    },
                    {"temperature": 0.7},
                    _trajectory(validate, step),
                    trace=False,
                )
                for validate, step, count in [(True, 0, 64), (False, 1, 1), (True, 5, 1)]
                for index in range(count)
            )
        )

    asyncio.run(run_batches())

    assert len(dispatch.calls) == len(dispatch.postprocessed) == 67
    assert sum(fields["global_steps"] == 0 for _, fields, _ in dispatch.calls) == 64
    assert len({id(context) for _, _, context in dispatch.calls}) == 67
    for _, fields, context in dispatch.calls:
        assert "validate" not in fields
        assert context.is_validation is (fields["global_steps"] != 1)
        assert context.step == fields["global_steps"]
    for validate, fields in dispatch.postprocessed:
        assert validate is (fields["global_steps"] != 1)
        assert "validate" not in fields
    finished = [entry for entry in dispatch.statuses if entry["tag"]["status"] == "finished"]
    assert len(finished) == 66
    assert all(entry["partition_id"] == ("train" if entry["key"].startswith("1-") else "val") for entry in finished)
    assert not any(entry["tag"]["status"] == "failure" for entry in dispatch.statuses)
