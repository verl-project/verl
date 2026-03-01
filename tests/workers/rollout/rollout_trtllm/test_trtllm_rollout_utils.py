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
import asyncio
import uuid

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

UNIMODAL_MODEL_PATH = "Qwen/Qwen2.5-Math-7B"
MULTIMODAL_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

MAX_MODEL_LEN = 4096
RESPONSE_LENGTH = 256
MAX_NUM_SEQS = 16
GPU_MEMORY_UTILIZATION = 0.8
TENSOR_PARALLEL_SIZE = 1


def create_test_image(width: int = 224, height: int = 224) -> Image.Image:
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img_array[i, j] = [
                int(255 * i / height),
                int(255 * j / width),
                int(255 * (i + j) / (height + width)),
            ]
    return Image.fromarray(img_array)


def create_rollout_config_dict():
    config_dict = {
        "_target_": "verl.workers.config.RolloutConfig",
        "name": "trtllm",
        "mode": "async",
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "do_sample": True,
        "n": 1,
        "prompt_length": 512,
        "response_length": RESPONSE_LENGTH,
        "dtype": "bfloat16",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "ignore_eos": False,
        "enforce_eager": True,
        "free_cache_engine": False,
        "data_parallel_size": 1,
        "tensor_model_parallel_size": TENSOR_PARALLEL_SIZE,
        "pipeline_model_parallel_size": 1,
        "max_num_batched_tokens": 8192,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "load_format": "auto",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
    }
    return OmegaConf.create(config_dict)


def create_model_config_dict(model_path: str):
    config_dict = {
        "_target_": "verl.workers.config.HFModelConfig",
        "path": model_path,
        "trust_remote_code": True,
        "load_tokenizer": True,
    }
    return OmegaConf.create(config_dict)


def get_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def get_processor(model_path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestUnimodalTRTLLMRollout:
    @pytest.fixture(scope="class")
    def ray_context(self):
        if ray.is_initialized():
            ray.shutdown()
        ray.init(ignore_reinit_error=True)
        yield
        ray.shutdown()

    @pytest.fixture(scope="class")
    def trtllm_replica(self, ray_context):
        from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica

        rollout_config = create_rollout_config_dict()
        model_config = create_model_config_dict(UNIMODAL_MODEL_PATH)

        replica = TRTLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=torch.cuda.device_count(),
            is_reward_model=False,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(replica.init_standalone())

        yield replica

        loop.close()

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return get_tokenizer(UNIMODAL_MODEL_PATH)

    @pytest.mark.parametrize(
        "prompt",
        [
            "What is 2 + 2?",
            "Solve for x: 3x + 5 = 20",
            "Calculate the derivative of x^2 + 3x + 1",
        ],
    )
    def test_unimodal_generate(self, trtllm_replica, tokenizer, prompt):
        replica = trtllm_replica

        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(text, return_tensors="pt")[0].tolist()

        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "logprobs": True,
        }

        request_id = str(uuid.uuid4())
        output = ray.get(
            replica.server_handle.generate.remote(
                prompt_ids=input_ids,
                sampling_params=sampling_params,
                request_id=request_id,
            )
        )

        assert output is not None
        assert hasattr(output, "token_ids")
        assert len(output.token_ids) > 0

        generated_text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
        print("\n[Unimodal Test]")
        print(f"Prompt: {prompt}")
        print(f"Generated ({len(output.token_ids)} tokens): {generated_text[:300]}...")

    def test_unimodal_batch_generate(self, trtllm_replica, tokenizer):
        replica = trtllm_replica

        prompts = [
            "What is 1 + 1?",
            "What is 2 * 3?",
            "What is 10 / 2?",
        ]

        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "logprobs": False,
        }

        results = []

        for i, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(text, return_tensors="pt")[0].tolist()

            output = ray.get(
                replica.server_handle.generate.remote(
                    prompt_ids=input_ids,
                    sampling_params=sampling_params,
                    request_id=str(uuid.uuid4()),
                )
            )
            results.append(output)

        assert len(results) == len(prompts)
        for i, (prompt, result) in enumerate(zip(prompts, results, strict=False)):
            assert result is not None
            assert len(result.token_ids) > 0
            generated = tokenizer.decode(result.token_ids, skip_special_tokens=True)
            print(f"\n[Batch {i}] Prompt: {prompt}")
            print(f"Generated: {generated[:100]}...")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestMultimodalTRTLLMRollout:
    @pytest.fixture(scope="class")
    def ray_context(self):
        if ray.is_initialized():
            ray.shutdown()
        ray.init(ignore_reinit_error=True)
        yield
        ray.shutdown()

    @pytest.fixture(scope="class")
    def trtllm_vlm_replica(self, ray_context):
        from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica

        rollout_config = create_rollout_config_dict()
        model_config = create_model_config_dict(MULTIMODAL_MODEL_PATH)

        replica = TRTLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=torch.cuda.device_count(),
            is_reward_model=False,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(replica.init_standalone())

        yield replica

        loop.close()

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return get_tokenizer(MULTIMODAL_MODEL_PATH)

    @pytest.fixture(scope="class")
    def processor(self):
        return get_processor(MULTIMODAL_MODEL_PATH)

    @pytest.mark.parametrize(
        "prompt",
        [
            "Describe this image in detail.",
            "What colors do you see in this image?",
            "What patterns are visible in this image?",
        ],
    )
    def test_multimodal_generate_with_image(self, trtllm_vlm_replica, processor, tokenizer, prompt):
        replica = trtllm_vlm_replica

        test_image = create_test_image(224, 224)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print("text: ", text)
        input_ids = processor.tokenizer(text, return_tensors="pt", padding=True)["input_ids"][0].tolist()

        print(
            "input_ids decoded: ",
            processor.tokenizer.decode(input_ids, skip_special_tokens=False, add_special_tokens=False),
        )

        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "logprobs": False,
        }

        output = ray.get(
            replica.server_handle.generate.remote(
                prompt_ids=input_ids,
                sampling_params=sampling_params,
                request_id=str(uuid.uuid4()),
                image_data=[test_image],
            )
        )

        assert output is not None
        assert hasattr(output, "token_ids")
        assert len(output.token_ids) > 0

        generated_text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
        print("\n[Multimodal Test]")
        print(f"Prompt: {prompt}")
        print(f"Image size: {test_image.size}")
        print(f"Generated ({len(output.token_ids)} tokens): {generated_text[:300]}...")

    @pytest.mark.parametrize(
        "image_size",
        [(224, 224), (384, 384), (512, 512)],
    )
    def test_multimodal_different_image_sizes(self, trtllm_vlm_replica, processor, tokenizer, image_size):
        replica = trtllm_vlm_replica

        width, height = image_size
        test_image = create_test_image(width, height)

        prompt = "What is shown in this image?"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = processor.tokenizer(text, return_tensors="pt", padding=True)["input_ids"][0].tolist()

        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "logprobs": False,
        }

        output = ray.get(
            replica.server_handle.generate.remote(
                prompt_ids=input_ids,
                sampling_params=sampling_params,
                request_id=str(uuid.uuid4()),
                image_data=[test_image],
            )
        )

        assert output is not None
        assert len(output.token_ids) > 0
        print(f"\n[Image Size {image_size}] Generated {len(output.token_ids)} tokens")

    def test_multimodal_text_only_fallback(self, trtllm_vlm_replica, tokenizer):
        replica = trtllm_vlm_replica

        prompt = "What is the capital of China?"
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(text, return_tensors="pt")[0].tolist()

        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "logprobs": False,
        }

        output = ray.get(
            replica.server_handle.generate.remote(
                prompt_ids=input_ids,
                sampling_params=sampling_params,
                request_id=str(uuid.uuid4()),
            )
        )

        assert output is not None
        assert len(output.token_ids) > 0

        generated_text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
        print("\n[Text-only on VLM]")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestTRTLLMServerLifecycle:
    @pytest.fixture(scope="class")
    def ray_context(self):
        if ray.is_initialized():
            ray.shutdown()
        ray.init(ignore_reinit_error=True)
        yield
        ray.shutdown()

    @pytest.fixture(scope="class")
    def trtllm_replica_lifecycle(self, ray_context):
        from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica

        rollout_config = create_rollout_config_dict()
        model_config = create_model_config_dict(UNIMODAL_MODEL_PATH)

        replica = TRTLLMReplica(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=torch.cuda.device_count(),
            is_reward_model=False,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(replica.init_standalone())

        yield replica, loop

        loop.close()

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return get_tokenizer(UNIMODAL_MODEL_PATH)

    def test_wake_sleep_cycle(self, trtllm_replica_lifecycle, tokenizer):
        replica, loop = trtllm_replica_lifecycle

        prompt = "Hello, world!"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(text, return_tensors="pt")[0].tolist()

        sampling_params = {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "logprobs": False}

        output1 = ray.get(
            replica.server_handle.generate.remote(
                prompt_ids=input_ids,
                sampling_params=sampling_params,
                request_id=str(uuid.uuid4()),
            )
        )
        assert output1 is not None
        assert len(output1.token_ids) > 0
        print(f"\n[Before Sleep] Generated {len(output1.token_ids)} tokens")

        loop.run_until_complete(replica.sleep())
        print("[Sleep] Server put to sleep")

        loop.run_until_complete(replica.wake_up())
        print("[Wake Up] Server woken up")

        output2 = ray.get(
            replica.server_handle.generate.remote(
                prompt_ids=input_ids,
                sampling_params=sampling_params,
                request_id=str(uuid.uuid4()),
            )
        )
        assert output2 is not None
        assert len(output2.token_ids) > 0
        print(f"[After Wake Up] Generated {len(output2.token_ids)} tokens")
