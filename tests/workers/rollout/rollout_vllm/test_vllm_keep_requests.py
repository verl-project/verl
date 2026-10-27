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
"""Two-GPU correctness test: live MoE requests survive real NCCL/IPC weight updates."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3MoeConfig

from verl.checkpoint_engine import CheckpointEngineManager, CheckpointEngineRegistry
from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import split_resource_pool
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.rollout import MultiTurnConfig
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.router import GlobalRequestLoadBalancer
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


def _digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class KeepRequestObserver(vLLMColocateWorkerExtension):
    """Observe real forward positions and cache storage without changing scheduling."""

    def install_keep_observer(self):
        self.keep_trace = []
        self.keep_steps = []
        execute = self.model_runner.execute_model

        def record(scheduler_output, *args, **kwargs):
            self.keep_steps.append(list(scheduler_output.num_scheduled_tokens))
            cached = scheduler_output.scheduled_cached_reqs
            starts = dict(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
            starts.update({r.req_id: r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs})
            self.keep_trace.extend(
                (req_id, starts[req_id], count) for req_id, count in scheduler_output.num_scheduled_tokens.items()
            )
            return execute(scheduler_output, *args, **kwargs)

        self.model_runner.execute_model = record

    def keep_snapshot(self):
        runner = self.model_runner
        return {
            "trace": list(self.keep_trace),
            "steps": list(self.keep_steps),
            "requests": {
                key: {
                    "blocks": [
                        table.gpu[index, : runner.block_tables.num_blocks.np[group, index]].tolist()
                        for group, table in enumerate(runner.block_tables.block_tables)
                    ],
                    "computed": int(runner.req_states.num_computed_tokens.gpu[index]),
                }
                for key, index in runner.req_states.req_id_to_index.items()
            },
            "kv": [(_digest(cache), cache.data_ptr()) for cache in runner.kv_caches],
            "weights": {
                name: _digest(param)
                for name, param in runner.model.named_parameters()
                if name in ("model.embed_tokens.weight", "lm_head.weight", "model.layers.0.mlp.gate.weight")
            },
        }


class ObservedServer(vLLMHttpServer):
    """Expose snapshots only to the test, keeping the production lifecycle intact."""

    def _get_worker_extension_cls(self):
        return f"{__name__}.KeepRequestObserver"

    async def observed_requests(self):
        result = {}
        for req_id, state in self.engine.output_processor.request_states.items():
            tokens = list(state.detokenizer.output_token_ids)
            result[state.external_req_id] = {
                "internal_id": req_id,
                "tokens": tokens,
                "logprobs": [
                    row[token].logprob for row, token in zip(state.logprobs_processor.logprobs, tokens, strict=False)
                ],
                "routing": np.concatenate(state.routed_experts_chunks) if state.routed_experts_chunks else None,
            }
        return result


class ObservedToolServer(ObservedServer):
    async def generate(self, *args, **kwargs):
        output = await super().generate(*args, **kwargs)
        self.turns.append(
            {
                "session_id": kwargs["session_id"],
                "request_id": kwargs["request_id"],
                "input_salt": kwargs.get("cache_salt"),
                "prompt_tokens": len(kwargs["prompt_ids"]),
                "output_tokens": len(output.token_ids),
                "stop_reason": output.stop_reason,
                **output.extra_fields,
            }
        )
        return output

    async def observed_turns(self):
        return self.turns

    async def init_turn_observer(self):
        self.turns = []


class OneToolCallParser:
    """Make a random model take one tool turn without changing its generated tokens."""

    stop_token_ids = []

    async def extract_tool_calls(self, response_ids, tools):
        return "", [FunctionCall(name="echo", arguments="{}")]


class ObservedReplica(vLLMReplica):
    """Capture the real paused state before and after the manager transfers weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(ObservedServer)
        self.boundaries = []
        self.observe = False

    async def pause_generation(self):
        await super().pause_generation()
        if self.observe:
            self.before = await self.server_handle.collective_rpc.remote("keep_snapshot")
            self.frontend_before = await self.server_handle.observed_requests.remote()

    async def resume_generation(self):
        if self.observe:
            after = await self.server_handle.collective_rpc.remote("keep_snapshot")
            self.boundaries.append((self.before, after, self.frontend_before))
        await super().resume_generation()


class WeightSender(Worker):
    """Real checkpoint sender with a tiny model and optional next-token SGD updates."""

    def __init__(self, model_path, checkpoint_config):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        self.checkpoint_engine = CheckpointEngineRegistry.new(
            "nccl",
            bucket_size=checkpoint_config.update_weights_bucket_megabytes << 20,
            is_master=True,
            multi_sender=False,
        )
        self.change_weights = False
        self.updates = []

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def enable_updates(self):
        self.change_weights = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps=None, mode=None):
        if self.change_weights:
            self.optimizer.zero_grad()
            tokens = torch.tensor([[1, 4, 5, 6, 7, 8]], device="cuda")
            loss = self.model(tokens, labels=tokens).loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.updates.append((float(loss.detach()), float(grad_norm)))
        await self.checkpoint_engine.send_weights(self.model.named_parameters(), global_steps=global_steps)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_results(self):
        return self.updates


def _save_tiny_moe(path):
    torch.manual_seed(17)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=2048,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    model.save_pretrained(path)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0", pad_token="t0", bos_token="t1", eos_token="t2"
    ).save_pretrained(path)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires separate sender and rollout GPUs")
@pytest.mark.parametrize("prefix_cache_mode", ["versioned", "shared", "disabled"])
def test_keep_requests_across_weight_updates(tmp_path, prefix_cache_mode):
    model_path = tmp_path / "tiny_moe"
    _save_tiny_moe(model_path)
    pythonpath = os.pathsep.join((os.getcwd(), str(Path(__file__).resolve().parent)))
    ray.init(address="local", num_gpus=2, num_cpus=12, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
    try:
        asyncio.run(_exercise_keep(str(model_path), prefix_cache_mode))
    finally:
        ray.shutdown()


async def _exercise_keep(model_path, prefix_cache_mode):
    checkpoint = CheckpointEngineConfig(backend="nccl", update_weights_bucket_megabytes=16)
    config = RolloutConfig(
        name="vllm",
        mode="async",
        tensor_model_parallel_size=1,
        prompt_length=64,
        response_length=256,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enforce_eager=True,
        preserve_kv_cache=True,
        enable_prefix_caching=prefix_cache_mode != "disabled",
        prefix_cache_mode="shared" if prefix_cache_mode == "shared" else "versioned",
        enable_rollout_routing_replay=True,
        calculate_log_probs=True,
        free_cache_engine=False,
        gpu_memory_utilization=0.2,
        checkpoint_engine=checkpoint,
        engine_kwargs={
            "vllm": {
                "kv_cache_memory_bytes": 8 << 20,
                "moe_backend": "triton",
                "attention_backend": "TRITON_ATTN",
            }
        },
    )
    pool = RayResourcePool(process_on_nodes=[2], max_colocate_count=2)
    sender_pool, rollout_pool = split_resource_pool(pool, [1, 1])
    sender = RayWorkerGroup(
        resource_pool=sender_pool,
        ray_cls_with_init=RayClassWithInitArgs(
            cls=ray.remote(WeightSender),
            model_path=model_path,
            checkpoint_config=checkpoint,
        ),
    )
    replica = ObservedReplica(0, config, HFModelConfig(path=model_path), gpus_per_node=2)
    await replica.init_standalone(rollout_pool)
    manager = CheckpointEngineManager(checkpoint, sender, [replica])
    await manager.update_weights(global_steps=0)
    await replica.server_handle.collective_rpc.remote("install_keep_observer")
    prompt = [1, 4, 5, 6] * 16
    sampling = {"temperature": 0.0, "max_tokens": 256, "ignore_eos": True, "logprobs": True}
    reference = await replica.server_handle.generate.remote(prompt, dict(sampling), "reference")
    # Establish repeatability before using the uninterrupted output as an oracle.
    repeat = await replica.server_handle.generate.remote(prompt, dict(sampling), "repeat")
    assert (repeat.extra_fields["num_cached_tokens"] > 0) == config.enable_prefix_caching
    assert repeat.token_ids == reference.token_ids
    assert repeat.log_probs == reference.log_probs
    np.testing.assert_array_equal(repeat.routed_experts, reference.routed_experts)
    summaries = []
    for changed, start_version in ((False, 0), (True, 2)):
        if changed:
            sender.enable_updates()
        request_id = "changed" if changed else "unchanged"
        replica.observe = True
        replica.boundaries.clear()
        pending = replica.server_handle.generate.remote(prompt, dict(sampling), request_id)
        for offset, target in ((1, 8), (2, 96)):
            async with asyncio.timeout(60):
                while True:
                    active = await replica.server_handle.observed_requests.remote()
                    if request_id in active and len(active[request_id]["tokens"]) >= target:
                        break
                    await asyncio.sleep(0.01)
            await manager.update_weights(global_steps=start_version + offset)
        output = await pending
        replica.observe = False
        assert output.stop_reason == "completed"
        assert len(output.token_ids) == len(output.log_probs) == 256
        assert output.routed_experts.shape == (len(prompt) + 255, 2, 2)
        assert output.extra_fields["min_global_steps"] == start_version
        assert output.extra_fields["max_global_steps"] == start_version + 2
        internal_ids = []
        boundary_tokens = []
        for before, after, frontend in replica.boundaries:
            prefix = frontend[request_id]
            internal_ids.append(prefix["internal_id"])
            boundary_tokens.append(len(prefix["tokens"]))
            for pre, post in zip(before, after, strict=True):
                assert pre["trace"] == post["trace"], "forward ran while weights were being updated"
                assert pre["requests"] == post["requests"], "request position/block ownership changed"
                assert pre["kv"] == post["kv"], "weight update changed the KV contents or allocation"
                assert (pre["weights"] != post["weights"]) == changed
            count = len(prefix["tokens"])
            assert output.token_ids[:count] == prefix["tokens"]
            assert output.log_probs[:count] == prefix["logprobs"]
            np.testing.assert_array_equal(output.routed_experts[: len(prefix["routing"])], prefix["routing"])
        assert len(set(internal_ids)) == 1
        states = await replica.server_handle.collective_rpc.remote("keep_snapshot")
        positions = [
            position
            for req_id, start, count in states[0]["trace"]
            if req_id == internal_ids[0]
            for position in range(start, start + count)
        ]
        cached = output.extra_fields["num_cached_tokens"]
        assert positions == list(range(cached, len(prompt) + 255)), "history was dropped or re-prefilled"
        if not changed:
            assert output.token_ids == reference.token_ids
            assert output.log_probs == reference.log_probs
            np.testing.assert_array_equal(output.routed_experts, reference.routed_experts)
        next_turn = await replica.server_handle.generate.remote(
            prompt + output.token_ids[:16],
            {**sampling, "max_tokens": 1},
            f"{request_id}-next-turn",
            cache_salt=output.extra_fields.get("cache_salt"),
        )
        fresh = await replica.server_handle.generate.remote(
            prompt, {**sampling, "max_tokens": 1}, f"{request_id}-fresh"
        )
        fresh_repeat = await replica.server_handle.generate.remote(
            prompt, {**sampling, "max_tokens": 1}, f"{request_id}-fresh-repeat"
        )
        assert (next_turn.extra_fields["num_cached_tokens"] > 0) == config.enable_prefix_caching
        assert (fresh.extra_fields["num_cached_tokens"] > 0) == (prefix_cache_mode == "shared")
        assert (fresh_repeat.extra_fields["num_cached_tokens"] > 0) == config.enable_prefix_caching
        summaries.append(
            {
                "changed": changed,
                "pause_output_lengths": boundary_tokens,
                "forward_positions": len(positions),
                "recomputed_positions": len(positions) - len(set(positions)),
                "cached_tokens": cached,
                "next_turn_cached_tokens": next_turn.extra_fields["num_cached_tokens"],
                "fresh_cached_tokens": fresh.extra_fields["num_cached_tokens"],
                "fresh_repeat_cached_tokens": fresh_repeat.extra_fields["num_cached_tokens"],
            }
        )
    updates = sender.update_results()[0]
    assert len(updates) == 2
    assert all(np.isfinite(loss) and np.isfinite(grad) and grad > 0 for loss, grad in updates)
    print(
        "KEEP_REQUEST_RESULTS="
        + json.dumps({"prefix_cache_mode": prefix_cache_mode, "requests": summaries, "optimizer_updates": updates})
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires separate sender and rollout GPUs")
def test_concurrent_tool_loops_across_weight_updates(tmp_path):
    model_path = tmp_path / "tiny_moe"
    _save_tiny_moe(model_path)
    pythonpath = os.pathsep.join((os.getcwd(), str(Path(__file__).resolve().parent)))
    ray.init(address="local", num_gpus=2, num_cpus=12, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
    try:
        asyncio.run(_exercise_tool_loops(str(model_path)))
    finally:
        ray.shutdown()


async def _exercise_tool_loops(model_path):
    checkpoint = CheckpointEngineConfig(backend="nccl", update_weights_bucket_megabytes=16)
    config = RolloutConfig(
        name="vllm",
        mode="async",
        tensor_model_parallel_size=1,
        prompt_length=128,
        response_length=768,
        max_model_len=1024,
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        preserve_kv_cache=True,
        calculate_log_probs=True,
        free_cache_engine=False,
        gpu_memory_utilization=0.2,
        checkpoint_engine=checkpoint,
        multi_turn=MultiTurnConfig(enable=True, max_assistant_turns=2),
        engine_kwargs={
            "vllm": {
                "kv_cache_memory_bytes": 8 << 20,
                "moe_backend": "triton",
                "attention_backend": "TRITON_ATTN",
            }
        },
    )
    pool = RayResourcePool(process_on_nodes=[2], max_colocate_count=2)
    sender_pool, rollout_pool = split_resource_pool(pool, [1, 1])
    sender = RayWorkerGroup(
        resource_pool=sender_pool,
        ray_cls_with_init=RayClassWithInitArgs(
            cls=ray.remote(WeightSender), model_path=model_path, checkpoint_config=checkpoint
        ),
    )
    replica = ObservedReplica(0, config, HFModelConfig(path=model_path), gpus_per_node=2)
    replica.server_class = ray.remote(ObservedToolServer)
    await replica.init_standalone(rollout_pool)
    manager = CheckpointEngineManager(checkpoint, sender, [replica])
    await manager.update_weights(global_steps=0)
    await replica.server_handle.collective_rpc.remote("install_keep_observer")
    await replica.server_handle.init_turn_observer.remote()
    load_balancer = ray.remote(GlobalRequestLoadBalancer).remote({"rollout": replica.server_handle})
    trainer_config = OmegaConf.create({"actor_rollout_ref": {"rollout": OmegaConf.structured(config)}, "data": {}})
    client = FullyAsyncLLMServerClient(trainer_config, load_balancer)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    tokenizer.backend_tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.chat_template = (
        "{% for message in messages %}{{ 't3 ' + message['content'] + ' t2 ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 't4 ' }}{% endif %}"
    )
    tool = FunctionTool(
        name="echo",
        fn=lambda: "t5",
        tool_schema=OpenAIFunctionToolSchema.model_validate(
            {"type": "function", "function": {"name": "echo", "description": "Return a fixed response."}}
        ),
    )
    loop = ToolAgentLoop(
        trainer_config=DictConfigWrap(trainer_config),
        server_manager=client,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=None,
        data_config=DictConfigWrap(trainer_config.data),
        tools=ToolListWrap([tool]),
    )
    loop.tool_parser = OneToolCallParser()
    prompt = [{"role": "user", "content": " ".join(["t4", "t5", "t6", "t7"] * 16)}]
    sampling = {"temperature": 0.0, "max_tokens": 256, "ignore_eos": True, "logprobs": True}
    tasks = [asyncio.create_task(loop.run(dict(sampling), raw_prompt=prompt)) for _ in range(4)]
    sender.enable_updates()
    replica.observe = True
    overlaps = []
    for version, target in ((1, 8), (2, 96)):
        async with asyncio.timeout(60):
            while True:
                active = await replica.server_handle.observed_requests.remote()
                if len(active) == 4 and all(len(state["tokens"]) >= target for state in active.values()):
                    break
                await asyncio.sleep(0.01)
        overlaps.append({"version": version, "active_token_counts": [len(s["tokens"]) for s in active.values()]})
        await manager.update_weights(global_steps=version)
    outputs = await asyncio.gather(*tasks)
    replica.observe = False
    fresh = await loop.run(dict(sampling), raw_prompt=prompt)
    turns = await replica.server_handle.observed_turns.remote()
    snapshots = await replica.server_handle.collective_rpc.remote("keep_snapshot")
    max_scheduled = max(len(step) for step in snapshots[0]["steps"])
    updates = sender.update_results()[0]
    print(
        "TOOL_LOOP_RESULTS="
        + json.dumps(
            {
                "enable_prefix_caching": config.enable_prefix_caching,
                "prefix_cache_mode": config.prefix_cache_mode,
                "max_scheduled_concurrency": max_scheduled,
                "sync_overlap": overlaps,
                "turns": turns,
                "tasks": [
                    {
                        "num_turns": output.num_turns,
                        "generated_tokens": sum(output.response_mask),
                        **output.extra_fields,
                    }
                    for output in [*outputs, fresh]
                ],
                "optimizer_updates": updates,
            }
        )
    )
    assert config.enable_prefix_caching and config.prefix_cache_mode == "versioned"
    assert max_scheduled == 4
    assert len(updates) == 2
    assert all(np.isfinite(loss) and np.isfinite(grad) and grad > 0 for loss, grad in updates)
    for before, after, frontend in replica.boundaries:
        assert len(frontend) == 4
        for pre, post in zip(before, after, strict=True):
            assert pre["trace"] == post["trace"]
            assert pre["requests"] == post["requests"]
            assert pre["kv"] == post["kv"]
            assert pre["weights"] != post["weights"]
    sessions = {}
    for turn in turns:
        sessions.setdefault(turn["session_id"], []).append(turn)
    assert len(sessions) == 5
    for index, session_turns in enumerate(sessions.values()):
        first, second = session_turns
        expected_salt = "2" if index == 4 else "0"
        assert first["input_salt"] is None
        assert second["input_salt"] == first["cache_salt"] == second["cache_salt"] == expected_salt
        assert second["num_cached_tokens"] > 0
        assert second["min_global_steps"] == second["max_global_steps"] == 2
        for turn in session_turns:
            assert turn["output_tokens"] == 256
            assert turn["stop_reason"] == "completed"
        assert first["min_global_steps"] == (2 if index == 4 else 0)
        assert first["max_global_steps"] == 2
    assert list(sessions.values())[-1][0]["num_cached_tokens"] == 0
    for output in outputs:
        assert output.num_turns == 4
        assert sum(output.response_mask) == 512
        assert output.extra_fields["cache_salt"] == "0"
        assert output.extra_fields["min_global_steps"] == 0
        assert output.extra_fields["max_global_steps"] == 2
    assert fresh.extra_fields["cache_salt"] == "2"
    assert fresh.extra_fields["min_global_steps"] == fresh.extra_fields["max_global_steps"] == 2
