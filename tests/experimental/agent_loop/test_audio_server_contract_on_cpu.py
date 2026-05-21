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

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LLM_SERVER_SOURCE = REPO_ROOT / "verl/workers/rollout/llm_server.py"
FULLY_ASYNC_ROLLOUTER_SOURCE = REPO_ROOT / "verl/experimental/fully_async_policy/fully_async_rollouter.py"
VLLM_SERVER_SOURCE = REPO_ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"


def _load_module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_async_server_manager_generate_accepts_audio_and_mm_kwargs() -> None:
    module = _load_module_ast(LLM_SERVER_SOURCE)

    generate_fn = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "LLMServerClient":
            for inner in node.body:
                if isinstance(inner, ast.AsyncFunctionDef) and inner.name == "generate":
                    generate_fn = inner
                    break

    assert generate_fn is not None
    arg_names = [arg.arg for arg in generate_fn.args.kwonlyargs]
    assert "audio_data" in arg_names
    assert "mm_processor_kwargs" in arg_names


def test_async_server_manager_generate_forwards_audio_and_mm_kwargs() -> None:
    source = LLM_SERVER_SOURCE.read_text(encoding="utf-8")
    assert 'multimodal_kwargs["audio_data"] = audio_data' in source
    assert 'multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs' in source


def test_fully_async_server_manager_generate_forwards_audio_and_mm_kwargs() -> None:
    source = FULLY_ASYNC_ROLLOUTER_SOURCE.read_text(encoding="utf-8")
    assert "audio_data=audio_data" in source
    assert "mm_processor_kwargs=mm_processor_kwargs" in source


def test_vllm_generate_includes_audio_and_mm_processor_kwargs() -> None:
    source = VLLM_SERVER_SOURCE.read_text(encoding="utf-8")
    assert 'multi_modal_data["audio"] = audio_data' in source
    assert 'prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs' in source
