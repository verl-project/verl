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

"""CPU-only coverage for vLLM router-replay version compatibility."""

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging import version

SERVER_PATH = (
    Path(__file__).resolve().parents[4] / "verl" / "workers" / "rollout" / "vllm_rollout" / "vllm_async_server.py"
)


def _is_router_replay_block(node: ast.If) -> bool:
    return ast.unparse(node.test) == "self.config.enable_rollout_routing_replay"


def _load_router_replay_guard(installed_version: str):
    """Compile only the guard so this test does not need Ray or vLLM installed."""
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"), filename=str(SERVER_PATH))
    server_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMHttpServer")
    launch_server = next(
        node for node in server_class.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "launch_server"
    )
    replay_block = next(
        node for node in ast.walk(launch_server) if isinstance(node, ast.If) and _is_router_replay_block(node)
    )

    function = ast.FunctionDef(
        name="apply_router_replay_config",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="args")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[replay_block, ast.Return(value=ast.Name(id="args", ctx=ast.Load()))],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "_VLLM_VERSION": version.parse(installed_version),
        "Mapping": Mapping,
        "json": json,
        "version": version,
        "vllm": SimpleNamespace(__version__=installed_version),
    }
    exec(compile(module, str(SERVER_PATH), "exec"), namespace)
    return namespace["apply_router_replay_config"]


def _server(*, router_replay: bool):
    return SimpleNamespace(config=SimpleNamespace(enable_rollout_routing_replay=router_replay))


def test_router_replay_with_mtp_speculation_requires_vllm_026():
    apply_config = _load_router_replay_guard("0.25.1")

    with pytest.raises(
        RuntimeError,
        match=r"MTP speculative rollout with router replay requires vLLM >= 0\.26\.0 .*installed: 0\.25\.1",
    ):
        apply_config(_server(router_replay=True), {"speculative_config": {"method": "mtp"}})


def test_router_replay_without_mtp_rejects_vllm_before_022():
    apply_config = _load_router_replay_guard("0.21.1")

    with pytest.raises(
        RuntimeError,
        match=r"enable_rollout_routing_replay=True requires vLLM >= 0\.22\.0 .*installed: 0\.21\.1",
    ):
        apply_config(_server(router_replay=True), {})


def test_router_replay_without_mtp_accepts_vllm_022():
    apply_config = _load_router_replay_guard("0.22.0")

    assert apply_config(_server(router_replay=True), {}) == {"enable_return_routed_experts": True}


def test_combined_router_replay_reports_026_floor_before_022_floor():
    apply_config = _load_router_replay_guard("0.21.1")

    with pytest.raises(RuntimeError, match=r"MTP speculative rollout with router replay requires vLLM >= 0\.26\.0"):
        apply_config(_server(router_replay=True), {"speculative_config": {"method": "mtp"}})


def test_router_replay_with_mtp_speculation_accepts_vllm_026():
    apply_config = _load_router_replay_guard("0.26.0")

    args = {"speculative_config": {"method": "mtp"}}
    assert apply_config(_server(router_replay=True), args) == {
        "speculative_config": {"method": "mtp"},
        "enable_return_routed_experts": True,
    }


def test_engine_kwargs_mtp_speculation_is_guarded_after_normalization():
    apply_config = _load_router_replay_guard("0.25.1")

    with pytest.raises(RuntimeError, match=r"MTP speculative rollout with router replay requires vLLM >= 0\.26\.0"):
        apply_config(_server(router_replay=True), {"speculative_config": json.dumps({"method": "mtp"})})


def test_non_mtp_speculation_does_not_raise_026_floor():
    apply_config = _load_router_replay_guard("0.25.1")

    args = {"speculative_config": {"method": "eagle"}}
    assert apply_config(_server(router_replay=True), args)["enable_return_routed_experts"] is True


def test_mtp_speculation_without_router_replay_has_no_added_version_floor():
    apply_config = _load_router_replay_guard("0.21.1")

    args = {"speculative_config": {"method": "mtp"}}
    assert apply_config(_server(router_replay=False), args) == args
