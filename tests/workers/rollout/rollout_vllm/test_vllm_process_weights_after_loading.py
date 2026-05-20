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
E2E test for process_weights_after_loading.

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_process_weights_after_loading.py -v -s
    python tests/workers/rollout/rollout_vllm/test_vllm_process_weights_after_loading.py
"""

import asyncio
import gc
import os
import time
from uuid import uuid4

import ray
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.device import is_support_ipc
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutMode, TokenOutput
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

MODEL_ID_DEEPSEEK = os.environ.get("MODEL_ID_DEEPSEEK", "deepseek-ai/DeepSeek-V2-Lite-Chat")
MODEL_PATH_DEEPSEEK = os.environ.get("MODEL_PATH_DEEPSEEK", os.path.expanduser(f"~/.cache/models/{MODEL_ID_DEEPSEEK}"))
MODEL_ID_QWEN3_4B = os.environ.get("MODEL_ID_QWEN3_4B", "Qwen/Qwen3-4B")
MODEL_PATH_QWEN3_4B = os.environ.get("MODEL_PATH_QWEN3_4B", os.path.expanduser(f"~/.cache/models/{MODEL_ID_QWEN3_4B}"))


def _build_config(load_format: str, model_path: str):
    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "vllm",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 64,
            "max_model_len": 4096,
            "dtype": "bfloat16",
            "load_format": load_format,
            "enforce_eager": False,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "prompt_length": 512,
            "response_length": 256,
            "top_k": -1,
            "top_p": 1.0,
            "temperature": 0.0,
        }
    )
    model_cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.HFModelConfig",
            "path": model_path,
            "trust_remote_code": True,
            "load_tokenizer": True,
        }
    )
    return rollout_cfg, model_cfg


def _start_server(load_format: str, model_path: str, force_dummy: bool = False):
    runtime_env = {
        "TOKENIZERS_PARALLELISM": "true",
        "VERL_LOGGING_LEVEL": "INFO",
        "VLLM_LOGGING_LEVEL": "INFO",
        "HCCL_CONNECT_TIMEOUT": os.environ.get("HCCL_CONNECT_TIMEOUT", "1500"),
        "HCCL_HOST_SOCKET_PORT_RANGE": os.environ.get("HCCL_HOST_SOCKET_PORT_RANGE", "60000-60050"),
        "HCCL_NPU_SOCKET_PORT_RANGE": os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE", "61000-61050"),
    }
    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"env_vars": runtime_env})

    rollout_cfg, model_cfg = _build_config(load_format, model_path)
    server = (
        ray.remote(vLLMHttpServer)
        .options(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    "NCCL_CUMEM_ENABLE": "0",
                }
            },
            max_concurrency=16,
        )
        .remote(
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
    )

    if force_dummy:
        ray.get(server.__ray_call__.remote(lambda self: setattr(self.config, "load_format", "dummy")))

    ray.get(server.launch_server.remote())
    return server


def _iter_weights(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")
    try:
        yield from model.state_dict().items()
    finally:
        del model
        gc.collect()


def _update_weights(server, model_path: str):
    # Use the same zmq handle format as vLLMColocateWorkerExtension._get_zmq_handle()
    replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
    zmq_handle = f"ipc:///tmp/rl-colocate-zmq-replica-{replica_rank}-rank-0.sock"
    use_shm = not is_support_ipc()
    update_ref = server.collective_rpc.remote("update_weights_from_ipc", kwargs={"use_shm": use_shm})
    sender = BucketedWeightSender(zmq_handle=zmq_handle, bucket_size_mb=4096, use_shm=use_shm)
    asyncio.run(sender.async_send_weights(_iter_weights(model_path)))
    ray.get(update_ref, timeout=1800)


def _generate(server, prompt: str, tag: str, model_path: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_ids = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
    )
    output = ray.get(
        server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params={"max_tokens": 2046, "temperature": 0.0, "top_p": 1.0, "top_k": -1},
            request_id=f"test_{tag}_{uuid4().hex[:8]}",
        ),
        timeout=300,
    )
    assert isinstance(output, TokenOutput) and len(output.token_ids) > 0
    text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
    assert text.strip() != ""
    return text


def _clear_npu_memory():
    gc.collect()
    time.sleep(2)


def _run_compare_test(model_path: str, prompt: str, model_name: str):
    dummy_server = None
    auto_server = None
    try:
        dummy_server = _start_server("dummy", model_path, force_dummy=True)
        _update_weights(dummy_server, model_path)
        ray.get(dummy_server.set_global_steps.remote(1))
        dummy_text = _generate(dummy_server, prompt, "dummy", model_path)

        ray.kill(dummy_server)
        dummy_server = None
        _clear_npu_memory()

        auto_server = _start_server("auto", model_path, force_dummy=False)
        auto_text = _generate(auto_server, prompt, "auto", model_path)

        print(f"\n[{model_name}] Prompt: {prompt}\n[dummy+update] {dummy_text}\n[auto] {auto_text}\n")
        assert dummy_text == auto_text, (
            f"{model_name} outputs mismatch:\n[dummy+update] {dummy_text}\n[auto] {auto_text}"
        )
    finally:
        if dummy_server:
            ray.kill(dummy_server)
        if auto_server:
            ray.kill(auto_server)
        if ray.is_initialized():
            ray.shutdown()


def test_compare_dummy_update_and_auto_outputs_same_prompt():
    """Test ACL graph mode (npugraph_ex) with DeepSeek-V2-Lite-Chat model."""
    _run_compare_test(
        MODEL_PATH_DEEPSEEK,
        prompt="write a poem about the moon.",
        model_name="DeepSeek-V2-Lite-Chat",
    )
    