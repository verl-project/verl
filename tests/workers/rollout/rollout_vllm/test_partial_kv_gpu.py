# Copyright 2026 Individual Contributor: HaitaoWuTJU
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
"""Two-GPU, four-request full-pool partial KV preservation regression.

This tests actual CuMem copies, native weight reload, graphs and continuation.
The update case uses a small HF Adam actor, not the full V1 training loop.
Checkpoint recovery and production-scale training are outside this test.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")


@pytest.mark.parametrize("optimizer_update", [False, True], ids=["same_weights", "colocated_adam"])
def test_partial_kv_full_pool_native_roundtrip(tmp_path, optimizer_update):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Partial KV GPU regression requires two CUDA GPUs")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    from verl.workers.rollout.vllm_rollout.partial_kv import PartialKVOffload

    torch.manual_seed(17)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
    model.save_pretrained(tmp_path)
    del model

    async def run():
        args = AsyncEngineArgs(
            model=str(tmp_path),
            tokenizer=str(tmp_path),
            skip_tokenizer_init=True,
            tensor_parallel_size=2,
            dtype="bfloat16",
            max_model_len=256,
            max_num_batched_tokens=64,
            max_num_seqs=4,
            kv_cache_memory_bytes=64 * 1024**2,
            enable_sleep_mode=True,
            enable_prefix_caching=True,
            enforce_eager=False,
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
            worker_extension_cls="verl.workers.rollout.vllm_rollout.partial_kv_worker.PartialKVWorkerExtension",
        )
        llm = AsyncLLM.from_engine_args(args)
        owner = SimpleNamespace(
            engine=llm,
            global_steps=0,
            config=SimpleNamespace(tensor_model_parallel_size=2, partial_rollout_kv_offload_headroom_gb=0),
            _submission_paused=False,
            _admitting=0,
            _rejecting=False,
            _resume_event=asyncio.Event(),
        )
        owner._resume_event.set()
        ready = asyncio.Event()
        progressed = set()
        latest = {}

        async def collect(prompt, sampling_params, request_id, **kwargs):
            final = None
            async for output in llm.generate(prompt, sampling_params, request_id):
                final = output
                latest[request_id] = output
                if output.outputs and len(output.outputs[0].token_ids) >= 16:
                    progressed.add(request_id)
                    if len(progressed) == 4:
                        ready.set()
            return final

        owner._generate_admitted = collect
        lifecycle = PartialKVOffload(owner)
        lifecycle.install()

        def request(index, prefix):
            return lifecycle.generate(
                prompt={"prompt_token_ids": [8 + index, 12 + index, 20 + index]},
                sampling_params=SamplingParams(temperature=0, max_tokens=192, ignore_eos=True, logprobs=5),
                request_id=f"{prefix}-{index}",
            )

        async def start_batch(prefix):
            # Identical admission/prefill shapes in both arms. Independent
            # arrival order can otherwise change a BF16 greedy trajectory
            # before the offload boundary being tested.
            await llm.pause_generation(mode="keep", clear_cache=False)
            batch = [asyncio.create_task(request(i, prefix)) for i in range(4)]
            async with asyncio.timeout(30):
                while llm.get_num_unfinished_requests() != 4:
                    await asyncio.sleep(0.001)
            await llm.resume_generation()
            return batch

        tasks = []
        try:
            await llm.engine_core.set_weight_version_async("0")
            tasks = await start_batch("baseline")
            baseline = await asyncio.gather(*tasks)
            await llm.reset_prefix_cache()
            progressed.clear()
            ready.clear()
            tasks = await start_batch("partial")
            await asyncio.wait_for(ready.wait(), timeout=60)
            await lifecycle.pause(True)
            assert all(not task.done() for task in tasks), "Fixture must pause four genuinely in-flight requests"
            identities = await llm.collective_rpc("partial_kv_stats")
            assert len({row["uuid"] for row in identities}) == 2
            assert all(row["cgroup"] == Path("/proc/self/cgroup").read_text() for row in identities)
            prefixes = [list(latest[f"partial-{i}"].outputs[0].token_ids) for i in range(4)]
            prefix_logps = [
                [row[token].logprob for row, token in zip(latest[f"partial-{i}"].outputs[0].logprobs, ids, strict=True)]
                for i, ids in enumerate(prefixes)
            ]
            before = await llm.collective_rpc("partial_kv_hashes")
            weights_before = await llm.collective_rpc("compute_weight_checksums")
            await lifecycle.sleep()
            offload_receipts = lifecycle.receipts
            assert all(
                row["pools"]["kv_cache"]["cpu_backup_bytes"] == row["pools"]["kv_cache"]["allocated_bytes"] > 0
                for row in lifecycle.receipts
            )
            if optimizer_update:
                # Real Adam on a GPU shared with the sleeping TP2 engine.
                # This is a small HF actor, not a full V1/Megatron trainer test.
                actor = Qwen2ForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).to("cuda:0")
                original = {name: parameter.detach().cpu().clone() for name, parameter in actor.named_parameters()}
                optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)
                tokens = torch.tensor([[8, 12, 20, 21, 22, 23, 24]], device="cuda:0")
                actor(input_ids=tokens, labels=tokens).loss.backward()
                optimizer.step()
                assert any(
                    not torch.equal(parameter.detach().cpu(), original[name])
                    for name, parameter in actor.named_parameters()
                )
                optimizer.zero_grad(set_to_none=True)
                actor.to(device="cpu", dtype=torch.bfloat16).save_pretrained(tmp_path)
                del optimizer, actor, tokens, original
                torch.accelerator.synchronize()
                torch.accelerator.empty_cache()
            await lifecycle.wake(["weights"])
            # Native reload writes the complete original model back into the
            # discarded weight allocations, without rebuilding request state.
            await llm.collective_rpc("reload_weights")
            weights_after = await llm.collective_rpc("compute_weight_checksums")
            if optimizer_update:
                assert all(old != new for old, new in zip(weights_before, weights_after, strict=True))
            else:
                assert weights_before == weights_after
            await lifecycle.commit_version(1)
            owner.global_steps = 1
            await lifecycle.wake(["kv_cache"])
            assert before == await llm.collective_rpc("partial_kv_hashes")
            await lifecycle.resume()
            resumed = await asyncio.wait_for(asyncio.gather(*tasks), timeout=120)
            for index, ((expected, _), (actual, segments)) in enumerate(zip(baseline, resumed, strict=True)):
                expected, actual = expected.outputs[0], actual.outputs[0]
                assert actual.token_ids[: len(prefixes[index])] == prefixes[index]
                assert segments[0]["policy_version"] == 0 and segments[-1]["policy_version"] == 1
                assert sum(segment["length"] for segment in segments) == len(actual.token_ids)
                expected_logps = [
                    row[token].logprob for row, token in zip(expected.logprobs, expected.token_ids, strict=False)
                ]
                actual_logps = [
                    row[token].logprob for row, token in zip(actual.logprobs, actual.token_ids, strict=False)
                ]
                assert actual_logps[: len(prefix_logps[index])] == prefix_logps[index]
                assert torch.isfinite(torch.tensor(actual_logps)).all()
                if not optimizer_update:
                    assert actual.token_ids == expected.token_ids
                    torch.testing.assert_close(
                        torch.tensor(actual_logps), torch.tensor(expected_logps), atol=2e-3, rtol=2e-3
                    )
            print(
                "partial_kv_validation="
                + json.dumps(
                    dict(
                        optimizer_update=optimizer_update,
                        requests=len(resumed),
                        gpu_uuids=[row["uuid"] for row in identities],
                        pool_bytes=[row["pools"] for row in offload_receipts],
                        prefix_lengths=[len(ids) for ids in prefixes],
                        version_segments=[segments for _, segments in resumed],
                        kv_and_runtime_hashes_equal=True,
                        native_weights_changed=weights_before != weights_after,
                        main_container_cgroups_match=True,
                    )
                )
            )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            llm.shutdown()

    asyncio.run(run())
