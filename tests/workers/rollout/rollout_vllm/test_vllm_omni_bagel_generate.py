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
E2E test for BAGEL RL pipeline via vLLMOmniHttpServer.

Uses verl's rollout server with BAGEL's multi-stage pipeline
(thinker on GPU 0, DiT on GPU 1) and BagelPipelineWithLogProb.

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_omni_bagel_generate.py -v -s
"""

import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from verl.workers.rollout.replica import DiffusionOutput, RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

MODEL_PATH = Path(os.path.expanduser("~/models/tiny-random/BAGEL-7B-MoT"))
STAGE_CONFIG = os.environ.get("BAGEL_STAGE_CONFIG", "")  # TODO: Point to the location on the CI server by default

DEFAULT_PROMPT = (
    "a beautiful sunset over the ocean with vibrant orange and purple clouds reflecting on the calm water surface"
)


# ---------------------------------------------------------------------
#                👇 Test Helper Functions & Fixtures 👇
# ---------------------------------------------------------------------


def _tokenize_prompt(text: str) -> list[int]:
    """Tokenize a text prompt into token IDs for BAGEL."""
    from transformers import AutoTokenizer

    from verl.utils.tokenizer import normalize_token_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    token_ids = normalize_token_ids(tokenizer.encode(text))
    return token_ids


@pytest.fixture
def init_server():
    """Create and launch a vLLMOmniHttpServer Ray actor with BAGEL."""
    if not STAGE_CONFIG:
        pytest.skip("BAGEL_STAGE_CONFIG env var not set")

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
            "gpu_memory_utilization": 0.9,
            "max_num_batched_tokens": 32768,
            "max_num_seqs": 1,
            "max_model_len": 32768,
            "dtype": "bfloat16",
            "load_format": "auto",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "n": 1,
            "num_inference_steps": 10,
            "engine_kwargs": {
                "vllm_omni": {
                    "custom_pipeline": "examples.flowgrpo_trainer.vllm_omni.pipeline_bagel.BagelPipelineWithLogProb",
                    "stage_configs_path": STAGE_CONFIG,
                }
            },
        }
    )

    model_cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.DiffusionModelConfig",
            "path": MODEL_PATH,
            "architecture": "OmniBagelForConditionalGeneration",
            "trust_remote_code": True,
            "load_tokenizer": False,
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
        gpus_per_node=2,
        nnodes=1,
        cuda_visible_devices="0,1",
    )

    ray.get(server.launch_server.remote())

    yield server

    ray.shutdown()


# ---------------------------------------------------------------------
#                          👇 Tests 👇
# ---------------------------------------------------------------------


def test_generate(init_server):
    """generate() returns a valid DiffusionOutput with CHW image in [0, 1]."""
    server = init_server

    request_id = f"test_{uuid4().hex[:8]}"
    output = ray.get(
        server.generate.remote(
            prompt_ids=_tokenize_prompt(DEFAULT_PROMPT),
            sampling_params={
                "num_inference_steps": 10,
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
    """generate() with SDE scheduler returns non-empty log_probs and RL artifacts."""
    server = init_server

    request_id = f"test_lp_{uuid4().hex[:8]}"
    output = ray.get(
        server.generate.remote(
            prompt_ids=_tokenize_prompt(DEFAULT_PROMPT),
            sampling_params={
                "num_inference_steps": 10,
                "noise_level": 0.7,
                "sde_type": "sde",
                "logprobs": True,
            },
            request_id=request_id,
        ),
        timeout=300,
    )

    assert isinstance(output, DiffusionOutput)
    assert len(output.diffusion_output) == 3

    lp = output.log_probs
    assert lp is not None, "log_probs should be present when logprobs=True"
    print(f"log_probs: shape={getattr(lp, 'shape', len(lp))}")

    extra = output.extra_fields
    assert extra.get("all_latents") is not None, "all_latents should be present"
    assert extra.get("all_timesteps") is not None, "all_timesteps should be present"
    print(f"all_latents: shape={getattr(extra['all_latents'], 'shape', len(extra['all_latents']))}")
    print(f"all_timesteps: shape={getattr(extra['all_timesteps'], 'shape', len(extra['all_timesteps']))}")


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
            sampling_params={"num_inference_steps": 10},
            request_id=rid,
        )
        refs.append(ref)

    results = ray.get(refs, timeout=600)

    for i, res in enumerate(results):
        assert isinstance(res, DiffusionOutput), f"Request {i}: expected DiffusionOutput"
        assert len(res.diffusion_output) == 3, f"Request {i}: expected 3 channels"
        assert res.stop_reason in ("completed", "aborted", None)

    print(f"All {n_requests} concurrent requests returned valid DiffusionOutput")


# ---------------------------------------------------------------------
#                     👇 LoRA helpers 👇
# ---------------------------------------------------------------------

# Tiny BAGEL: hidden_size=64, 2 Q heads, 2 KV heads, head_dim=32
# QKV packed dim = (2+2+2)*32 = 192
_LORA_DIM = 64
_LORA_QKV_DIM = 192
_LORA_MODULE = "bagel.language_model.model.layers.0.self_attn.qkv_proj"
_LORA_RANK = 4


def _make_synthetic_lora(adapter_dir: Path):
    """Create a synthetic rank-4 LoRA adapter on disk."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(42)
    lora_a = torch.randn((_LORA_RANK, _LORA_DIM), dtype=torch.float32, generator=gen) * 0.1
    lora_b = torch.randn((_LORA_QKV_DIM, _LORA_RANK), dtype=torch.float32, generator=gen) * 0.5
    save_file(
        {
            f"base_model.model.{_LORA_MODULE}.lora_A.weight": lora_a,
            f"base_model.model.{_LORA_MODULE}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": _LORA_RANK, "lora_alpha": _LORA_RANK, "target_modules": [_LORA_MODULE]}),
        encoding="utf-8",
    )
    return str(adapter_dir)


def test_generate_with_lora(init_server):
    """LoRA adapter changes output and deactivation restores baseline."""
    from vllm_omni.lora.request import LoRARequest

    server = init_server

    with tempfile.TemporaryDirectory() as tmp_dir:
        lora_path = _make_synthetic_lora(Path(tmp_dir) / "bagel_lora")
        lora_request = LoRARequest(lora_name="test_lora", lora_int_id=42, lora_path=lora_path)

        # 1) Baseline (no LoRA)
        baseline = ray.get(
            server.generate.remote(
                prompt_ids=_tokenize_prompt(DEFAULT_PROMPT),
                sampling_params={"num_inference_steps": 10},
                request_id=f"lora_base_{uuid4().hex[:8]}",
            ),
            timeout=300,
        )

        # 2) With LoRA
        with_lora = ray.get(
            server.generate.remote(
                prompt_ids=_tokenize_prompt(DEFAULT_PROMPT),
                sampling_params={"num_inference_steps": 10},
                request_id=f"lora_on_{uuid4().hex[:8]}",
                lora_request=lora_request,
                lora_scale=1.0,
            ),
            timeout=300,
        )

        # 3) Deactivated (no LoRA again)
        restored = ray.get(
            server.generate.remote(
                prompt_ids=_tokenize_prompt(DEFAULT_PROMPT),
                sampling_params={"num_inference_steps": 10},
                request_id=f"lora_off_{uuid4().hex[:8]}",
            ),
            timeout=300,
        )

    assert isinstance(baseline, DiffusionOutput)
    assert isinstance(with_lora, DiffusionOutput)
    assert isinstance(restored, DiffusionOutput)

    base_arr = np.array(baseline.diffusion_output)
    lora_arr = np.array(with_lora.diffusion_output)

    diff_lora = np.abs(base_arr - lora_arr).mean()

    print(f"LoRA diff from baseline: {diff_lora:.4f}")

    # LoRA should visibly change output
    assert diff_lora > 0.001, f"LoRA had no effect: diff={diff_lora}"
    # Output is not corrupted
    assert diff_lora < 80, f"LoRA output looks corrupted: diff={diff_lora}"
