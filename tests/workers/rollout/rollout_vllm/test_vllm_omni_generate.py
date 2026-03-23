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

"""
E2E test for vLLMOmniHttpServer generate flow.

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py -v -s
    python tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py
"""

import os
from pathlib import Path
from uuid import uuid4

import pytest
import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import DiffusionOutput, RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

MODEL_PATH = Path(os.path.expanduser("~/models/tiny-random/Qwen-Image"))


# ---------------------------------------------------------------------
#                👇 Test Helper Functions & Fixtures 👇
# ---------------------------------------------------------------------

_MIN_PROMPT_TOKENS = 35


def _tokenize_prompt(text: str) -> list[int]:
    """Tokenize a text prompt into valid token IDs for the model."""
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, "tokenizer"), trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    assert len(token_ids) > _MIN_PROMPT_TOKENS, (
        f"Prompt too short ({len(token_ids)} tokens, need >{_MIN_PROMPT_TOKENS}). "
        f"The pipeline drops the first 34 chat‑template prefix tokens; "
        f"use a longer prompt so content tokens remain after the drop."
    )
    return token_ids


@pytest.fixture
def init_server():
    """Create and launch a vLLMOmniHttpServer Ray actor with Qwen/Qwen-Image."""
    model_path = MODEL_PATH

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
            }
        },
        ignore_reinit_error=True,
    )

    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 256,
            "max_model_len": 1058,
            "dtype": "bfloat16",
            "load_format": "auto",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "n": 4,
            "height": 512,
            "width": 512,
            "num_inference_steps": 10,
            "guidance_scale": 4.0,
            "engine_kwargs": {
                "vllm_omni": {
                    "custom_pipeline": "examples.vllm_omni.pipeline_qwenimage.QwenImagePipelineWithLogProb",
                }
            },
        }
    )

    model_cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.DiffusionModelConfig",
            "path": model_path,
            "tokenizer_path": os.path.join(model_path, "tokenizer"),
            "trust_remote_code": True,
            "load_tokenizer": True,
        }
    )

    ServerCls = ray.remote(vLLMOmniHttpServer)
    server = ServerCls.options(
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
            }
        },
        max_concurrency=10,
    ).remote(
        config=rollout_cfg,
        model_config=model_cfg,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=1,
        nnodes=1,
        cuda_visible_devices="0",
    )

    ray.get(server.launch_server.remote())

    yield server

    ray.shutdown()


def test_generate(init_server):
    """generate() returns a valid DiffusionOutput with CHW image in [0, 1]."""
    server = init_server
    prompt_ids = _tokenize_prompt(
        "a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline"
    )

    request_id = f"test_{uuid4().hex[:8]}"
    output = ray.get(
        server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params={
                "num_inference_steps": 10,
                "guidance_scale": 4.0,
                "height": 512,
                "width": 512,
            },
            request_id=request_id,
        ),
        timeout=300,
    )

    assert isinstance(output, DiffusionOutput)
    assert len(output.diffusion_output) == 3, f"Expected 3 channels (CHW), got {len(output.diffusion_output)}"
    h, w = len(output.diffusion_output[0]), len(output.diffusion_output[0][0])
    assert h > 0 and w > 0
    assert output.stop_reason in ("completed", "aborted", None)

    # spot-check pixel range
    assert 0.0 <= output.diffusion_output[0][0][0] <= 1.0

    print(f"image: C=3 H={h} W={w}  stop_reason={output.stop_reason}")


def test_generate_with_logprobs(init_server):
    """generate() with logprobs=True returns log_probs list."""
    server = init_server
    prompt_ids = _tokenize_prompt(
        "a futuristic city at night with neon lights glowing on tall glass "
        "skyscrapers and flying vehicles soaring between the buildings"
    )

    request_id = f"test_lp_{uuid4().hex[:8]}"
    output = ray.get(
        server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params={
                "num_inference_steps": 10,
                "guidance_scale": 4.0,
                "height": 512,
                "width": 512,
                "logprobs": True,
            },
            request_id=request_id,
        ),
        timeout=300,
    )

    assert isinstance(output, DiffusionOutput)
    assert len(output.diffusion_output) == 3
    assert output.log_probs is not None, "log_probs should be present when logprobs=True"
    assert isinstance(output.log_probs, list)
    assert len(output.log_probs) > 0

    print(f"log_probs: {len(output.log_probs)} values, sample: {output.log_probs[:3]}")


def test_generate_concurrent(init_server):
    """Multiple concurrent generate() calls all return valid DiffusionOutput."""
    server = init_server
    n_requests = 4

    prompts = [
        "a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline",
        "a fluffy orange cat sitting on a wooden windowsill looking outside at "
        "a garden full of colorful flowers on a bright sunny afternoon",
        "a majestic mountain landscape covered with fresh white snow under a "
        "clear blue sky with pine trees in the foreground and a frozen lake",
        "a futuristic city at night with neon lights glowing on tall glass "
        "skyscrapers and flying vehicles soaring between the buildings",
    ]

    refs = []
    for i in range(n_requests):
        rid = f"concurrent_{i}_{uuid4().hex[:8]}"
        ref = server.generate.remote(
            prompt_ids=_tokenize_prompt(prompts[i]),
            sampling_params={
                "num_inference_steps": 10,
                "guidance_scale": 4.0,
                "height": 512,
                "width": 512,
            },
            request_id=rid,
        )
        refs.append(ref)

    results = ray.get(refs, timeout=600)

    for i, res in enumerate(results):
        assert isinstance(res, DiffusionOutput), f"Request {i}: expected DiffusionOutput"
        assert len(res.diffusion_output) == 3, f"Request {i}: expected 3 channels"
        assert res.stop_reason in ("completed", "aborted", None)

    print(f"All {n_requests} concurrent requests returned valid DiffusionOutput")
