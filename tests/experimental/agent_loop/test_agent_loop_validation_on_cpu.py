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

"""Exercise native dispatch methods without importing Ray or GPU runtimes."""

from __future__ import annotations

import ast
import asyncio
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_methods(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name in names]
    assert sorted(node.name for node in methods) == sorted(names)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def dispatch():
    calls, postprocessed, statuses = [], [], []
    expected_output = object()

    class CustomAgent:
        async def run(self, sampling_params, **kwargs):
            await asyncio.sleep(0)
            calls.append((sampling_params, kwargs))
            return expected_output

    async def put_status(**kwargs):
        statuses.append(kwargs)

    namespace = {
        "asyncio": asyncio,
        "logger": logging.getLogger(__name__),
        "hydra": SimpleNamespace(utils=SimpleNamespace(instantiate=lambda **kwargs: CustomAgent())),
        "rollout_trace_attr": lambda **kwargs: nullcontext(),
        "_agent_loop_registry": {"custom": {}},
        "DictConfigWrap": lambda config: config,
        "ToolListWrap": lambda tools: tools,
        "tq": SimpleNamespace(async_kv_put=put_status),
    }
    _load_methods(
        REPO_ROOT / "verl/experimental/agent_loop/agent_loop.py",
        {"_run_agent_loop"},
        namespace,
    )
    _load_methods(
        REPO_ROOT / "verl/trainer/ppo/v1/agent_loop_tq.py",
        {"_run_prompt", "_settle_session_tasks"},
        namespace,
    )

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
        worker=Worker(), calls=calls, postprocessed=postprocessed, statuses=statuses, output=expected_output
    )


def _trajectory(validate, step):
    return {"validate": validate, "step": step, "sample_index": 0, "rollout_n": 0}


@pytest.mark.parametrize(("validate", "step"), [(False, 1), (True, 0), (True, 5)])
@pytest.mark.parametrize("dataset_mode", ["absent", False, True, "dataset-value"])
def test_agent_loop_receives_authoritative_validation_mode(dispatch, validate, step, dataset_mode):
    sampling_params = {"temperature": 0.7}
    fields = {"raw_prompt": ["question"], "global_steps": step, "custom_field": "preserved"}
    if dataset_mode != "absent":
        fields["validate"] = dataset_mode
    original_fields = dict(fields)
    result = asyncio.run(
        dispatch.worker._run_agent_loop(
            sampling_params, _trajectory(validate, step), agent_name="custom", trace=False, **fields
        )
    )

    assert result is dispatch.output
    assert fields == original_fields
    expected_fields = {key: value for key, value in fields.items() if key != "validate"}
    assert dispatch.calls == [(sampling_params, {**expected_fields, "validate": validate})]
    assert dispatch.postprocessed == [(validate, expected_fields)]
    assert sampling_params == {"temperature": 0.7}


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
    assert sum(fields["global_steps"] == 0 for _, fields in dispatch.calls) == 64
    for _, fields in dispatch.calls:
        assert fields["validate"] is (fields["global_steps"] != 1)
    for validate, fields in dispatch.postprocessed:
        assert validate is (fields["global_steps"] != 1)
        assert "validate" not in fields
    finished = [entry for entry in dispatch.statuses if entry["tag"]["status"] == "finished"]
    assert len(finished) == 66
    assert all(entry["partition_id"] == ("train" if entry["key"].startswith("1-") else "val") for entry in finished)
    assert not any(entry["tag"]["status"] == "failure" for entry in dispatch.statuses)
