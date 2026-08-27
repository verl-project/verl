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
"""Unit tests for vllm max_num_batched_tokens vs max_model_len handling.

Regression guard: with enable_chunked_prefill=False, vllm >= 0.24 raises
``ValueError`` when ``max_num_batched_tokens < max_model_len``
(``verify_max_model_len`` in ``vllm/config/scheduler.py``). verl's default
``max_num_batched_tokens`` (8192) is below every modern model's
``max_position_embeddings``, so the vLLM HTTP server raises
``max_num_batched_tokens`` to ``max_model_len`` before engine creation.
"""

from types import SimpleNamespace

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def _make_engine_args(
    max_num_batched_tokens=8192, max_model_len=None, enable_chunked_prefill=False
) -> AsyncEngineArgs:
    return AsyncEngineArgs(
        model="dummy",
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        tensor_parallel_size=1,
        worker_extension_cls="",
    )


def _hf_config(max_position_embeddings=131072):
    return SimpleNamespace(max_position_embeddings=max_position_embeddings)


class TestEnsureMaxNumBatchedTokensValid:
    def test_chunked_off_raises_when_below_model_len(self):
        """8192 < max_position_embeddings with chunked off -> raised to model len."""
        args = _make_engine_args()
        vLLMHttpServer._ensure_max_num_batched_tokens_valid(args, _hf_config())
        assert args.max_num_batched_tokens == 131072

    def test_chunked_off_respects_explicit_max_model_len(self):
        """Explicit max_model_len (2048) already satisfies the check; unchanged."""
        args = _make_engine_args(max_model_len=2048)
        vLLMHttpServer._ensure_max_num_batched_tokens_valid(args, _hf_config())
        assert args.max_num_batched_tokens == 8192

    def test_chunked_off_raised_to_explicit_max_model_len(self):
        """Explicit max_model_len (131072) becomes the raised target."""
        args = _make_engine_args(max_num_batched_tokens=16384, max_model_len=131072)
        vLLMHttpServer._ensure_max_num_batched_tokens_valid(args, _hf_config())
        assert args.max_num_batched_tokens == 131072

    def test_chunked_on_no_change(self):
        """Chunked prefill enabled does not trigger the validation, no change."""
        args = _make_engine_args(enable_chunked_prefill=True)
        vLLMHttpServer._ensure_max_num_batched_tokens_valid(args, _hf_config())
        assert args.max_num_batched_tokens == 8192

    def test_skips_when_hf_config_unavailable(self):
        """Cannot resolve max_model_len without hf_config; leave untouched."""
        args = _make_engine_args()
        vLLMHttpServer._ensure_max_num_batched_tokens_valid(args, None)
        assert args.max_num_batched_tokens == 8192

    def test_none_max_num_batched_tokens_skipped(self):
        args = _make_engine_args(max_num_batched_tokens=None)
        vLLMHttpServer._ensure_max_num_batched_tokens_valid(args, _hf_config())
        assert args.max_num_batched_tokens is None
