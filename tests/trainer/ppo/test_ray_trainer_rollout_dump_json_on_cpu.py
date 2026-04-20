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

import ast
import json
import os
import tempfile
from pathlib import Path

import numpy as np


RAY_TRAINER_PATH = Path(__file__).resolve().parents[3] / "verl/trainer/ppo/ray_trainer.py"


def _load_rollout_dump_helper():
    source = RAY_TRAINER_PATH.read_text()
    module = ast.parse(source, filename=str(RAY_TRAINER_PATH))
    selected = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_jsonify_rollout_value"
    ]
    helper_module = ast.Module(body=selected, type_ignores=[])
    namespace = {"np": np}
    exec(compile(helper_module, filename=str(RAY_TRAINER_PATH), mode="exec"), namespace)
    return namespace["_jsonify_rollout_value"]


def _load_dump_generations():
    source = RAY_TRAINER_PATH.read_text()
    module = ast.parse(source, filename=str(RAY_TRAINER_PATH))
    helper_nodes = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_jsonify_rollout_value"
    ]
    dump_node = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "_dump_generations":
                    dump_node = child
                    break
    assert dump_node is not None
    helper_module = ast.Module(body=[*helper_nodes, dump_node], type_ignores=[])
    namespace = {"np": np, "json": json, "os": os}
    exec(compile(helper_module, filename=str(RAY_TRAINER_PATH), mode="exec"), namespace)
    return namespace["_dump_generations"]


def test_jsonify_rollout_value_converts_numpy_bool_scalar():
    jsonify_rollout_value = _load_rollout_dump_helper()

    value = jsonify_rollout_value(np.bool_(True))

    assert value is True
    assert isinstance(value, bool)


def test_jsonify_rollout_value_converts_nested_numpy_values():
    jsonify_rollout_value = _load_rollout_dump_helper()

    value = jsonify_rollout_value(
        {
            "keep": np.bool_(False),
            "scores": np.array([np.float32(1.5), np.float32(2.5)]),
            "nested": [np.int64(3), {"ok": np.bool_(True)}],
        }
    )

    assert value == {
        "keep": False,
        "scores": [1.5, 2.5],
        "nested": [3, {"ok": True}],
    }


def test_jsonify_rollout_value_produces_json_serializable_entry():
    jsonify_rollout_value = _load_rollout_dump_helper()

    entry = {
        "reward_flags": np.bool_(True),
        "metadata": {"accepted": np.bool_(False), "shape": np.array([1, 2])},
    }

    serialized = json.dumps(jsonify_rollout_value(entry), ensure_ascii=False)

    assert json.loads(serialized) == {
        "reward_flags": True,
        "metadata": {"accepted": False, "shape": [1, 2]},
    }


def test_dump_generations_writes_jsonl_with_numpy_bool_entries():
    dump_generations = _load_dump_generations()

    class DummyTrainer:
        global_steps = 7

    with tempfile.TemporaryDirectory() as tmpdir:
        dump_generations(
            DummyTrainer(),
            inputs=["prompt"],
            outputs=["answer"],
            gts=["gt"],
            scores=[1.0],
            reward_extra_infos_dict={
                "accepted": [np.bool_(True)],
                "nested": [{"keep": np.bool_(False), "shape": np.array([1, 2])}],
            },
            dump_path=tmpdir,
        )

        payload = Path(tmpdir, "7.jsonl").read_text().strip()

    assert json.loads(payload) == {
        "input": "prompt",
        "output": "answer",
        "gts": "gt",
        "score": 1.0,
        "step": 7,
        "accepted": True,
        "nested": {"keep": False, "shape": [1, 2]},
    }
