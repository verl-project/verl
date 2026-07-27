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
"""Routing-record merging across partial rollout resumes in FullyAsyncLLMServerClient.

Every backend normalizes ``routed_experts`` to numpy, but dtype and writability still
differ and the merge has to survive all of them:

- vLLM: writable ``np.uint8`` (``np.uint16`` above 256 experts)
- SGLang with ``skip_tokenizer_init=False``: read-only ``np.int32``, since
  ``extract_routed_experts_from_meta_info`` builds it via ``np.frombuffer`` over bytes
- SGLang with ``skip_tokenizer_init=True``: writable ``np.int32``, converted by
  ``async_sglang_server`` from the scheduler's CPU tensor
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from verl.workers.rollout import llm_server
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import TokenOutput

LAYERS, TOPK = 2, 4
BACKENDS = ["vllm", "sglang_detokenized", "sglang_skip_tokenizer_init"]


def _routing(markers: list[int], backend: str) -> np.ndarray:
    """Build a ``[len(markers), LAYERS, TOPK]`` routing record as ``backend`` delivers it.

    Every entry of row ``i`` is ``markers[i]`` so :func:`_markers` can recover which
    source rows survived the merge.
    """
    array = np.repeat(np.asarray(markers, dtype=np.int32), LAYERS * TOPK).reshape(-1, LAYERS, TOPK)
    if backend == "vllm":
        return array.astype(np.uint8)
    if backend == "sglang_detokenized":
        return np.frombuffer(array.tobytes(), dtype=np.int32).reshape(-1, LAYERS, TOPK)
    if backend == "sglang_skip_tokenizer_init":
        # Mirrors the ``.numpy()`` conversion async_sglang_server applies to the tensor.
        return torch.from_numpy(array).numpy()
    raise AssertionError(f"unknown backend: {backend}")


def _markers(routing: Any) -> list[int]:
    return [int(v) for v in routing[:, 0, 0]]


def _install_segments(monkeypatch: pytest.MonkeyPatch, segments: list[TokenOutput]) -> list[list[int]]:
    """Script the upstream so each resume attempt returns the next segment.

    ``FullyAsyncLLMServerClient.generate`` reaches the upstream through ``super()``, which
    binds to the class rather than the instance, so the base class has to be patched.
    Returns a list that accumulates the prompt each attempt was issued with.
    """
    pending = iter(segments)
    prompts_seen: list[list[int]] = []

    async def fake_generate(self, request_id, *, prompt_ids, **kwargs):
        prompts_seen.append(list(prompt_ids))
        return next(pending)

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fake_generate)

    real_sleep = asyncio.sleep

    async def no_wait(_delay, *args, **kwargs):
        return await real_sleep(0, *args, **kwargs)

    # The resume loop backs off 1s between attempts; collapse it to keep the test fast.
    monkeypatch.setattr(llm_server.asyncio, "sleep", no_wait)

    return prompts_seen


def _client() -> FullyAsyncLLMServerClient:
    # A config without ``async_training`` leaves partial rollout enabled, so the client
    # retries on "aborted" instead of returning after the first attempt.
    return FullyAsyncLLMServerClient(config=SimpleNamespace(), load_balancer_handle=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_resume_appends_only_newly_generated_routing(monkeypatch, backend):
    """Two resumes: each attempt re-sends the accumulated prompt and returns routing for
    prompt + response, so only the tail matching the fresh tokens may be appended."""
    segments = [
        TokenOutput(
            token_ids=[101, 102],
            routed_experts=_routing([10, 11, 12, 13, 14], backend),
            stop_reason="aborted",
        ),
        TokenOutput(
            token_ids=[103],
            routed_experts=_routing([20, 21, 22, 23, 24, 25], backend),
            stop_reason="aborted",
        ),
        TokenOutput(
            token_ids=[104, 105],
            routed_experts=_routing([30, 31, 32, 33, 34, 35, 36, 37], backend),
            stop_reason="completed",
        ),
    ]
    prompts_seen = _install_segments(monkeypatch, segments)

    output = await _client().generate(
        request_id="req-0",
        prompt_ids=[1, 2, 3],
        sampling_params={},
    )

    assert prompts_seen == [[1, 2, 3], [1, 2, 3, 101, 102], [1, 2, 3, 101, 102, 103]]
    assert output.token_ids == [101, 102, 103, 104, 105]
    # First attempt contributes all 5 of its rows; later attempts only their generated tail.
    assert _markers(output.routed_experts) == [10, 11, 12, 13, 14, 25, 36, 37]
    assert output.routed_experts.shape == (8, LAYERS, TOPK)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_merge_yields_numpy_and_preserves_dtype(monkeypatch, backend):
    """Backends agree on numpy, which is what lets the merge be a plain concatenate. Dtype
    still varies, and ``AgentLoopWorker._postprocess`` keys its padded buffer off it."""
    original = _routing([10, 11], backend)
    segments = [
        TokenOutput(token_ids=[101, 102], routed_experts=original, stop_reason="aborted"),
        TokenOutput(token_ids=[103], routed_experts=_routing([20, 21, 22], backend), stop_reason="completed"),
    ]
    _install_segments(monkeypatch, segments)

    merged = (await _client().generate(request_id="req-0", prompt_ids=[], sampling_params={})).routed_experts

    assert isinstance(merged, np.ndarray)
    assert merged.dtype == original.dtype


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_single_attempt_passes_backend_object_through(monkeypatch, backend):
    """Without a resume there is nothing to concatenate, so the backend object reaches the
    agent loop untouched -- including SGLang's read-only array."""
    original = _routing([10, 11], backend)
    segment = TokenOutput(token_ids=[101, 102], routed_experts=original, stop_reason="completed")
    _install_segments(monkeypatch, [segment])

    output = await _client().generate(request_id="req-0", prompt_ids=[], sampling_params={})

    assert output.routed_experts is original


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_merged_routing_converts_for_downstream(monkeypatch, backend):
    """Both downstream sinks normalize to int64 tensors: ``AgentLoopOutput.as_dict`` via
    ``torch.tensor`` and ``AgentLoopWorker._postprocess`` via ``torch.from_numpy``."""
    segments = [
        TokenOutput(token_ids=[101], routed_experts=_routing([10, 11], backend), stop_reason="aborted"),
        TokenOutput(token_ids=[102], routed_experts=_routing([20, 21, 22], backend), stop_reason="completed"),
    ]
    _install_segments(monkeypatch, segments)

    merged = (await _client().generate(request_id="req-0", prompt_ids=[], sampling_params={})).routed_experts

    as_dict_sink = torch.tensor(merged, dtype=torch.int64)
    assert as_dict_sink.shape == (3, LAYERS, TOPK)
    assert _markers(as_dict_sink) == [10, 11, 22]

    # _postprocess only copies when the array is read-only; concatenation already produced
    # a fresh writable buffer, so from_numpy is safe without one.
    assert merged.flags.writeable
    assert _markers(torch.from_numpy(merged)) == [10, 11, 22]
