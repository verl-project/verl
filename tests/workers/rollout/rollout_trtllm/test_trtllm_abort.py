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
Test TRT-LLM abort_all_requests and resume_generation functionality.

Usage:
    pytest tests/workers/rollout/rollout_trtllm/test_trtllm_abort.py -v -s

Environment variables:
    TRTLLM_TEST_MODEL_PATH_ROOT: parent directory containing model weights (default: ~/models)
    TRTLLM_TEST_TP_SIZE: tensor parallel size (default: 2)
    TRTLLM_TEST_GPUS_PER_NODE: number of GPUs (default: 2)
"""

import os
import subprocess
import time
from uuid import uuid4


def test_trtllm_abort():
    # ==================== Configuration ====================
    model_root = os.path.expanduser(os.getenv("TRTLLM_TEST_MODEL_PATH_ROOT", "~/models"))
    MODEL_PATH = os.path.join(model_root, "Qwen/Qwen2.5-1.5B-Instruct")
    GPUS_PER_NODE = int(os.getenv("TRTLLM_TEST_GPUS_PER_NODE", "2"))
    TP_SIZE = int(os.getenv("TRTLLM_TEST_TP_SIZE", "2"))
    ABORT_DELAY = 0.5  # seconds to wait before aborting
    NUM_PROMPTS = 8

    print("=" * 60)
    print("TRT-LLM Abort / Resume Test")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {GPUS_PER_NODE}, TP Size: {TP_SIZE}")
    print(f"Abort Delay: {ABORT_DELAY}s")
    print("=" * 60)

    # ==================== Initialize Ray ====================
    print("\n[1] Initializing Ray...")
    import ray
    from ray.util import placement_group_table
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    os.environ.setdefault("TLLM_RAY_FORCE_LOCAL_CLUSTER", "1")
    ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

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
        config.actor_rollout_ref.rollout.name = "trtllm"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = TP_SIZE
        config.actor_rollout_ref.rollout.prompt_length = 512
        config.actor_rollout_ref.rollout.response_length = 512  # long enough to be aborted mid-flight

        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model

        # ==================== Create TRTLLMHttpServer ====================
        print("\n[3] Creating TRTLLMHttpServer (this may take a while)...")
        from verl.single_controller.ray import RayResourcePool
        from verl.workers.rollout.replica import RolloutMode
        from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMHttpServer

        resource_pool = RayResourcePool(
            process_on_nodes=[GPUS_PER_NODE],
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="test_abort",
        )
        pgs = resource_pool.get_placement_groups()
        bundle_indices = [list(range(TP_SIZE))]

        pg_data = placement_group_table(pgs[0])
        node_id = pg_data["bundles_to_node_id"][bundle_indices[0][0]]

        server = TRTLLMHttpServer.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
            name="trtllm_abort_test",
        ).remote(
            config=rollout_config,
            model_config=model_config,
            is_reward_model=False,
            rollout_mode=RolloutMode.COLOCATED,
            workers=[],
            replica_rank=0,
            max_colocate_count=1,
            pgs=pgs,
            bundle_indices=bundle_indices,
        )

        ray.get(server.launch_server.remote())
        print("Server launched.")

        # ==================== Load Tokenizer ====================
        print("\n[4] Loading tokenizer...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # ==================== Prepare Prompts ====================
        print("\n[5] Preparing prompts...")
        from verl.utils.tokenizer import normalize_token_ids

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
            prompt_ids = normalize_token_ids(
                tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            )
            all_prompt_ids.append(prompt_ids)
        print(f"Prepared {NUM_PROMPTS} prompts")

        sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "logprobs": False,
        }

        # ==================== Phase 1: Abort Test ====================
        print("\n" + "=" * 60)
        print("PHASE 1: Abort Test")
        print("=" * 60)

        print(f"\n   Starting {NUM_PROMPTS} concurrent generations...")
        generate_refs = []
        for i, prompt_ids in enumerate(all_prompt_ids):
            request_id = f"abort_test_{i}_{uuid4().hex[:8]}"
            ref = server.generate.remote(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=None,
            )
            generate_refs.append((i, request_id, ref))
            print(f"      Started request {i}: {request_id}")

        print(f"\n   Waiting {ABORT_DELAY}s before abort...")
        time.sleep(ABORT_DELAY)

        print("   Calling abort_all_requests...")
        abort_start = time.perf_counter()
        ray.get(server.abort_all_requests.remote())
        abort_time = time.perf_counter() - abort_start
        print(f"   Abort took: {abort_time * 1000:.2f}ms")

        print("\n   Waiting for all generations to resolve...")
        outputs = []
        for i, request_id, ref in generate_refs:
            try:
                output = ray.get(ref, timeout=10.0)
                outputs.append((i, request_id, output))
            except ray.exceptions.GetTimeoutError:
                print(f"      Request {i} TIMED OUT!")
                outputs.append((i, request_id, None))

        # Results
        print("\n" + "=" * 60)
        print("PHASE 1 RESULTS")
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
            else:
                completed_count += 1
                print(f"[{i}] {request_id}: COMPLETED ({output.stop_reason}, {len(output.token_ids)} tokens)")

        print(f"\nSummary: {aborted_count} aborted, {completed_count} completed, {timeout_count} timeout")

        assert timeout_count == 0, f"No requests should timeout, got {timeout_count}"
        assert aborted_count + completed_count == NUM_PROMPTS, "All requests should finish (aborted or completed)"
        assert abort_time < 1.0, f"Abort should be fast (< 1s), took {abort_time:.2f}s"
        # At least some requests should have been aborted with partial tokens
        if aborted_count > 0:
            partial_token_counts = [len(out.token_ids) for _, _, out in outputs if out and out.stop_reason == "aborted"]
            assert any(n > 0 for n in partial_token_counts), (
                f"At least one aborted request should have partial tokens, got counts: {partial_token_counts}"
            )
        print("Phase 1 assertions PASSED")

        # ==================== Phase 2: Resume Test ====================
        print("\n" + "=" * 60)
        print("PHASE 2: Resume Test")
        print("=" * 60)

        print("\n   Calling resume_generation...")
        ray.get(server.resume_generation.remote())
        print("   resume_generation() returned.")

        print("\n   Submitting 2 fresh requests after resume...")
        resume_refs = []
        for i in range(2):
            request_id = f"resume_test_{i}_{uuid4().hex[:8]}"
            ref = server.generate.remote(
                prompt_ids=all_prompt_ids[i],
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=None,
            )
            resume_refs.append((i, request_id, ref))
            print(f"      Started request {i}: {request_id}")

        resume_outputs = []
        for i, request_id, ref in resume_refs:
            try:
                output = ray.get(ref, timeout=60.0)
                resume_outputs.append((i, request_id, output))
                print(f"[{i}] {request_id}: {output.stop_reason} ({len(output.token_ids)} tokens)")
            except ray.exceptions.GetTimeoutError:
                resume_outputs.append((i, request_id, None))
                print(f"[{i}] {request_id}: TIMEOUT")

        for i, request_id, output in resume_outputs:
            assert output is not None, f"Post-resume request {request_id} timed out"
            assert output.stop_reason != "aborted", (
                f"Post-resume request {request_id} was aborted (resume_generation() not working)"
            )
            assert len(output.token_ids) > 0, f"Post-resume request {request_id} returned no tokens"

        # ==================== Phase 3: Partial Rollout Test ====================
        print("\n" + "=" * 60)
        print("PHASE 3: Partial Rollout Test")
        print("=" * 60)
        print("Re-submitting aborted requests with prompt_ids + partial_token_ids")

        # Collect aborted outputs that have partial tokens to continue from
        aborted_with_tokens = [
            (i, request_id, out)
            for i, request_id, out in outputs
            if out is not None and out.stop_reason == "aborted" and len(out.token_ids) > 0
        ]
        print(f"\n   Found {len(aborted_with_tokens)} aborted requests with partial tokens to continue")

        if aborted_with_tokens:
            partial_refs = []
            for i, orig_request_id, partial_output in aborted_with_tokens[:2]:  # test up to 2
                # Re-submit: prompt_ids + accumulated partial tokens (the partial rollout pattern)
                continued_prompt_ids = list(all_prompt_ids[i]) + list(partial_output.token_ids)
                request_id = f"partial_rollout_{i}_{uuid4().hex[:8]}"
                ref = server.generate.remote(
                    prompt_ids=continued_prompt_ids,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    image_data=None,
                )
                partial_refs.append((i, orig_request_id, partial_output.token_ids, request_id, ref))
                print(f"      Re-submitted request {i}: {len(partial_output.token_ids)} partial tokens + prompt")

            print("\n   Waiting for partial rollout continuations...")
            for i, orig_request_id, partial_token_ids, request_id, ref in partial_refs:
                try:
                    output = ray.get(ref, timeout=60.0)
                    total_tokens = len(partial_token_ids) + len(output.token_ids)
                    print(
                        f"[{i}] {request_id}: {output.stop_reason} "
                        f"(+{len(output.token_ids)} new tokens, {total_tokens} total)"
                    )
                    assert output is not None, f"Partial rollout continuation {request_id} timed out"
                    assert output.stop_reason != "aborted", (
                        f"Partial rollout continuation {request_id} was aborted again"
                    )
                    assert len(output.token_ids) > 0, (
                        f"Partial rollout continuation {request_id} returned no new tokens"
                    )
                except ray.exceptions.GetTimeoutError as err:
                    raise AssertionError(f"Partial rollout continuation {request_id} timed out") from err

            print("Phase 3 assertions PASSED")
        else:
            print("   No aborted requests with partial tokens (all completed before abort) — skipping Phase 3")
            print("   (Try reducing ABORT_DELAY or increasing response_length to force mid-flight aborts)")

        # ==================== Phase 4: clear_kv_cache Test ====================
        print("\n" + "=" * 60)
        print("PHASE 4: clear_kv_cache Test")
        print("=" * 60)

        print("\n   Calling clear_kv_cache...")
        ray.get(server.clear_kv_cache.remote())
        print("   clear_kv_cache() returned.")

        print("\n   Submitting 1 request after clear_kv_cache to verify generation still works...")
        request_id = f"post_clear_{uuid4().hex[:8]}"
        ref = server.generate.remote(
            prompt_ids=all_prompt_ids[0],
            sampling_params=sampling_params,
            request_id=request_id,
            image_data=None,
        )
        try:
            output = ray.get(ref, timeout=60.0)
            print(f"   {request_id}: {output.stop_reason} ({len(output.token_ids)} tokens)")
            assert output is not None, "Post-clear_kv_cache request timed out"
            assert output.stop_reason != "aborted", "Post-clear_kv_cache request was unexpectedly aborted"
            assert len(output.token_ids) > 0, "Post-clear_kv_cache request returned no tokens"
        except ray.exceptions.GetTimeoutError as err:
            raise AssertionError("Post-clear_kv_cache request timed out") from err
        print("Phase 4 assertions PASSED")

    finally:
        ray.shutdown()
        subprocess.run(["ray", "stop"], capture_output=True)


if __name__ == "__main__":
    test_trtllm_abort()
