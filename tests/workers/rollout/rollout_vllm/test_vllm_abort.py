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
"""
Test vLLM abort functionality.

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_abort.py -v -s
    or
    python tests/workers/rollout/rollout_vllm/test_vllm_abort.py
"""

import asyncio
import os
import time
from uuid import uuid4


def test_vllm_abort():
    # ==================== Configuration ====================
    MODEL_PATH = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")  # /root/models/Qwen/Qwen2.5-1.5B-Instruct
    GPUS_PER_NODE = 2
    TP_SIZE = 1
    ROLLOUT_NAME = "vllm"
    ABORT_DELAY = 0.5  # seconds to wait before aborting

    print("=" * 60)
    print("vLLM Abort Test")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {GPUS_PER_NODE}, TP Size: {TP_SIZE}")
    print(f"Abort Delay: {ABORT_DELAY}s")
    print("=" * 60)

    # ==================== Initialize Ray ====================
    print("\n[1] Initializing Ray...")
    import ray

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )

    try:
        # ==================== Create Config ====================
        print("\n[2] Creating config...")
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("verl/verl/trainer/config")
        if not os.path.exists(config_dir):
            config_dir = os.path.abspath("verl/trainer/config")

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="ppo_trainer")

        config.trainer.n_gpus_per_node = GPUS_PER_NODE
        config.trainer.nnodes = 1
        config.actor_rollout_ref.model.path = MODEL_PATH
        config.actor_rollout_ref.rollout.name = ROLLOUT_NAME
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = TP_SIZE
        config.actor_rollout_ref.rollout.prompt_length = 512
        config.actor_rollout_ref.rollout.response_length = 512  # Longer for abort test

        # ==================== Create Rollout Server ====================
        print("\n[3] Creating rollout server (this may take a while)...")
        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model

        rollout_server_class = get_rollout_replica_class(ROLLOUT_NAME)
        server = rollout_server_class(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=GPUS_PER_NODE,
        )

        asyncio.run(server.init_standalone())
        server_handle = server._server_handle
        print(f"Server address: {server._server_address}")

        # ==================== Load Tokenizer ====================
        print("\n[4] Loading tokenizer...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # ==================== Prepare Prompts ====================
        print("\n[5] Preparing prompts (to ensure generation takes time)...")
        NUM_PROMPTS = 8
        prompts = [
            "Write a very long story about a brave knight and dragon.",
            "Explain the history of the Roman Empire in great detail.",
            "Describe quantum computing and its applications thoroughly.",
            "Write an essay about climate change and its global effects.",
            "Who won the Champions League in 2019?",
            "Write a detailed analysis of Shakespeare's Hamlet.",
            "Describe the process of photosynthesis in plants.",
            "Write about the French Revolution and its consequences.",
        ]

        all_prompt_ids = []
        for prompt in prompts[:NUM_PROMPTS]:
            messages = [{"role": "user", "content": prompt}]
            prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            all_prompt_ids.append(prompt_ids)
        print(f"Prepared {NUM_PROMPTS} prompts")

        # ==================== Start Generations and Abort ====================
        print("\n[6] Starting generations and then aborting...")

        sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "logprobs": False,
        }

        # Start all generations concurrently
        print(f"\n   Starting {NUM_PROMPTS} generations...")
        generate_refs = []
        for i, prompt_ids in enumerate(all_prompt_ids):
            request_id = f"abort_test_{i}_{uuid4().hex[:8]}"
            ref = server_handle.generate.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
            )
            generate_refs.append((i, request_id, ref))
            print(f"      Started request {i}: {request_id}")

        # Wait before aborting
        print(f"\n   Waiting {ABORT_DELAY}s before abort...")
        time.sleep(ABORT_DELAY)

        # Call abort
        print("   Calling abort_all_requests...")
        abort_start = time.perf_counter()
        abort_result = ray.get(server_handle.abort_all_requests.remote())
        abort_time = time.perf_counter() - abort_start

        print(f"   Abort took: {abort_time * 1000:.2f}ms")
        print(f"   Abort result: {abort_result}")

        # Wait for all generations to finish
        print("\n   Waiting for all generations to complete...")
        outputs = []
        for i, request_id, ref in generate_refs:
            try:
                output = ray.get(ref, timeout=10.0)
                outputs.append((i, request_id, output))
            except ray.exceptions.GetTimeoutError:
                print(f"      Request {i} timed out!")
                outputs.append((i, request_id, None))

        # ==================== Print Results ====================
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        aborted_count = 0
        completed_count = 0
        timeout_count = 0

        for i, request_id, output in outputs:
            if output is None:
                timeout_count += 1
                print(f"[{i}] {request_id}: TIMEOUT")
            elif output.stop_reason == "aborted":
                aborted_count += 1
                print(f"[{i}] {request_id}: ABORTED ({len(output.token_ids)} tokens)")
                print(f"Partial Output: {tokenizer.decode(output.token_ids)}")
            else:
                completed_count += 1
                print(f"[{i}] {request_id}: COMPLETED ({output.stop_reason}, {len(output.token_ids)} tokens)")
                print(f"Full Output: {tokenizer.decode(output.token_ids)}")

        print(f"\nSummary: {aborted_count} aborted, {completed_count} completed, {timeout_count} timeout")

        print("\n" + "=" * 60)
        print(f"Abort result: {abort_result}")
        print("=" * 60)
        print("Abort test completed!")

        # Assertions for pytest
        assert timeout_count == 0, "No requests should timeout"
        assert aborted_count + completed_count == NUM_PROMPTS, "All requests should finish"
        assert "aborted_count" in abort_result, "Abort result should contain aborted_count"
        assert abort_time < 1.0, "Abort should be fast (< 1 second)"

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    # Can still run as standalone script
    test_vllm_abort()
