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
import asyncio
from pathlib import Path

import pytest

AGENT_LOOP_PATH = Path(__file__).parents[3] / "verl" / "experimental" / "agent_loop" / "agent_loop.py"
SGLANG_SERVER_PATH = (
    Path(__file__).parents[3] / "verl" / "workers" / "rollout" / "sglang_rollout" / "async_sglang_server.py"
)


def _load_agent_loop_functions(*names):
    module = ast.parse(AGENT_LOOP_PATH.read_text())
    selected_nodes = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names:
            selected_nodes.append(node)

    found_names = {node.name for node in selected_nodes}
    missing_names = set(names) - found_names
    if missing_names:
        raise AssertionError(f"agent_loop helper missing: {sorted(missing_names)}")

    ns = {"Any": object, "hashlib": __import__("hashlib")}
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), str(AGENT_LOOP_PATH), "exec"), ns)
    return tuple(ns[name] for name in names)


def _load_sglang_server_functions(*names):
    module = ast.parse(SGLANG_SERVER_PATH.read_text())
    selected_nodes = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names:
            selected_nodes.append(node)

    found_names = {node.name for node in selected_nodes}
    missing_names = set(names) - found_names
    if missing_names:
        raise AssertionError(f"SGLang server helper missing: {sorted(missing_names)}")

    ns = {"Any": object}
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), str(SGLANG_SERVER_PATH), "exec"), ns)
    return tuple(ns[name] for name in names)


def test_stable_sglang_sampling_seed_is_positive_int31_on_cpu():
    (stable_seed,) = _load_agent_loop_functions("_stable_sglang_sampling_seed")

    seed = stable_seed(sample_index=0, rollout_n=0, step=0, base_seed=42)

    assert 0 < seed <= 0x7FFFFFFF
    assert seed == stable_seed(sample_index=0, rollout_n=0, step=0, base_seed=42)
    assert seed != stable_seed(sample_index=0, rollout_n=1, step=0, base_seed=42)
    assert seed != stable_seed(sample_index=0, rollout_n=0, step=1, base_seed=42)


def test_sglang_sampling_seed_base_defaults_for_deterministic_inference():
    (seed_base,) = _load_agent_loop_functions("_get_sglang_sampling_seed_base")

    assert seed_base({"enable_deterministic_inference": True}) == 42
    assert seed_base({"enable_deterministic_inference": True, "random_seed": 7}) == 7
    assert seed_base({"random_seed": 7}) is None
    assert seed_base({}) is None


def test_global_trajectory_chunks_avoid_duplicate_sglang_seeds_across_workers_on_cpu():
    get_trajectory_info, split_trajectory_info, stable_seed = _load_agent_loop_functions(
        "get_trajectory_info",
        "_split_trajectory_info_for_chunks",
        "_stable_sglang_sampling_seed",
    )

    repeated_prompt_indices = [1, 1, 1, 1]
    chunk_sizes = [2, 2]

    buggy_chunks = [
        asyncio.run(get_trajectory_info(step=10, index=repeated_prompt_indices[:2], validate=False)),
        asyncio.run(get_trajectory_info(step=10, index=repeated_prompt_indices[2:], validate=False)),
    ]
    assert [[item["rollout_n"] for item in chunk] for chunk in buggy_chunks] == [[0, 1], [0, 1]]
    buggy_seeds = [
        stable_seed(sample_index=item["sample_index"], rollout_n=item["rollout_n"], step=item["step"], base_seed=42)
        for chunk in buggy_chunks
        for item in chunk
    ]
    assert len(set(buggy_seeds)) < len(buggy_seeds)

    trajectory_info = asyncio.run(get_trajectory_info(step=10, index=repeated_prompt_indices, validate=False))
    chunks = split_trajectory_info(trajectory_info, chunk_sizes=chunk_sizes)

    assert [[item["rollout_n"] for item in chunk] for chunk in chunks] == [[0, 1], [2, 3]]

    seeds = [
        stable_seed(sample_index=item["sample_index"], rollout_n=item["rollout_n"], step=item["step"], base_seed=42)
        for chunk in chunks
        for item in chunk
    ]
    assert len(set(seeds)) == len(seeds)


def test_sglang_server_normalizes_custom_engine_kwargs_on_cpu():
    (normalize,) = _load_sglang_server_functions("_normalize_sglang_engine_kwargs")

    assert normalize({}) is False
    assert normalize({"enable_deterministic_inference": True}) is True

    fsdp_kwargs = {"rl_on_policy_target": "fsdp"}
    assert normalize(fsdp_kwargs) is True
    assert fsdp_kwargs == {}

    with pytest.raises(ValueError, match="rl_on_policy_target"):
        normalize({"rl_on_policy_target": True})


def test_sglang_server_copies_engine_kwargs_before_mutation_on_cpu():
    module = ast.parse(SGLANG_SERVER_PATH.read_text())
    server_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "SGLangHttpServer"
    )
    launch_fn = next(
        node for node in server_class.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "launch_server"
    )

    engine_kwargs_assignment = next(
        node
        for node in launch_fn.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "engine_kwargs" for target in node.targets)
    )

    assert (
        ast.unparse(engine_kwargs_assignment.value)
        == "dict((self.config.get('engine_kwargs', {}) or {}).get('sglang', {}) or {})"
    )
