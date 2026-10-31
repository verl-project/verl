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
"""Two-GPU integration, launched from an FSDP environment without vLLM.

Set VLLM_ISOLATED_PYTHON to a matching, separately provisioned rollout Python and
VLLM_ISOLATED_MODEL_PATH to a small local HF causal LM. Both environments must
install this checkout. No packages or models are downloaded by this test.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ray
import torch

from tests.checkpoint_engine.test_utils import TrainingWorkerTest
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.config import CheckpointConfig
from verl.workers.config import (
    CheckpointEngineConfig,
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.rollout.replica import get_rollout_replica_class
from verl.workers.rollout.runtime import runtime_identity

pytestmark = pytest.mark.skipif(
    not os.environ.get("VLLM_ISOLATED_PYTHON") or not os.environ.get("VLLM_ISOLATED_MODEL_PATH"),
    reason="requires separately provisioned FSDP/vLLM environments and a local model",
)


class RuntimeTrainingWorker(TrainingWorkerTest):
    """Real FSDP2 engine/optimizer plus the production NCCL checkpoint sender."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_runtime(self):
        assert importlib.util.find_spec("vllm") is None
        return runtime_identity()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimize(self, token_ids: list[int]):
        ids = torch.tensor([token_ids], device="cuda")
        with self.engine.train_mode():
            self.engine.optimizer_zero_grad()
            output = self.engine.module(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            loss = torch.nn.functional.cross_entropy(output.logits[:, :-1].float().flatten(0, 1), ids[:, 1:].flatten())
            loss.backward()
            grad_norm = self.engine.optimizer_step()
            self.engine.optimizer_zero_grad()
        return {"loss": loss.item(), "grad_norm": float(grad_norm)}


@pytest.mark.asyncio
async def test_isolated_runtime_generate_train_sync_and_resume(tmp_path):
    assert importlib.util.find_spec("vllm") is None, "run this test from the training-only environment"
    assert torch.cuda.device_count() >= 2
    rollout_python = os.environ["VLLM_ISOLATED_PYTHON"]
    model_config = HFModelConfig(
        path=os.environ["VLLM_ISOLATED_MODEL_PATH"],
        use_remove_padding=False,
        enable_gradient_checkpointing=False,
        override_config={"attn_implementation": "sdpa"},
    )
    checkpoint_config = CheckpointEngineConfig(backend="nccl", update_weights_bucket_megabytes=32)
    config = RolloutConfig(
        name="vllm",
        tensor_model_parallel_size=1,
        nnodes=1,
        n_gpus_per_node=1,
        prompt_length=32,
        response_length=8,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        gpu_memory_utilization=0.2,
        enforce_eager=True,
        load_format="safetensors",
        free_cache_engine=False,
        calculate_log_probs=True,
        checkpoint_engine=checkpoint_config,
        runtime={"py_executable": rollout_python, "startup_timeout": 300.0},
    )
    ray.init(
        num_cpus=8,
        num_gpus=2,
        include_dashboard=False,
        runtime_env={
            "py_executable": sys.executable,
            "env_vars": {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"},
        },
    )
    replica = None
    actor_wg = None
    # pytest's default import mode gives this file a bare module name which
    # is not importable in a fresh Ray worker. Ship the test actor by value.
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    try:
        # A valid training interpreter with no vLLM must fail during preflight,
        # before it creates any checkpoint worker or server actor.
        bad_config = RolloutConfig(
            name="vllm",
            tensor_model_parallel_size=1,
            checkpoint_engine=checkpoint_config,
            runtime={"py_executable": sys.executable, "startup_timeout": 60.0},
        )
        bad_replica = get_rollout_replica_class("vllm")(0, bad_config, model_config, gpus_per_node=1)
        with pytest.raises(Exception, match="vllm|PackageNotFoundError"):
            await bad_replica.init_standalone()
        assert bad_replica.workers == bad_replica.servers == []

        worker_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=model_config,
            engine_config=FSDPEngineConfig(
                strategy="fsdp2",
                fsdp_size=1,
                use_torch_compile=False,
                forward_only=False,
                use_dynamic_bsz=False,
                micro_batch_size_per_gpu=1,
            ),
            optimizer_config=FSDPOptimizerConfig(lr=1e-2, total_training_steps=3, weight_decay=0.0),
            checkpoint_config=CheckpointConfig(),
        )
        actor_wg = RayWorkerGroup(
            resource_pool=RayResourcePool([1], max_colocate_count=2),
            ray_cls_with_init=RayClassWithInitArgs(
                ray.remote(RuntimeTrainingWorker), config=worker_config, checkpoint_engine_config=checkpoint_config
            ),
            device_name="cuda",
        )
        actor_wg.reset()
        assert all(report["prefix"] == sys.prefix for report in actor_wg.inspect_runtime())

        replica = get_rollout_replica_class("vllm")(0, config, model_config, gpus_per_node=1)
        await replica.init_standalone()
        expected_prefix = str(Path(rollout_python).absolute().parent.parent)
        assert all(report["prefix"] == expected_prefix for report in replica.runtime_reports)
        checkpoint_reports = ray.get(
            [worker.__ray_call__.remote(lambda self: runtime_identity()) for worker in replica.workers]
        )
        assert all(report["prefix"] == expected_prefix for report in checkpoint_reports)
        manager = CheckpointEngineManager(checkpoint_config, actor_wg, [replica])

        async def generate(request_id):
            return await replica.server_handle.generate.remote(
                prompt_ids=[1, 10, 11, 12],
                sampling_params={"temperature": 0.0, "max_tokens": 1, "logprobs": True},
                request_id=request_id,
            )

        await manager.update_weights(global_steps=0)
        before = await generate("before")
        assert before.token_ids and torch.isfinite(torch.tensor(before.log_probs)).all()
        training_ids = [1, 10, 11, 12, before.token_ids[0]]
        metrics = actor_wg.optimize(training_ids)[0]
        assert metrics["loss"] > 0 and metrics["grad_norm"] > 0
        await manager.update_weights(global_steps=1)
        after = await generate("after")
        assert after.extra_fields["global_steps"] == 1
        assert before.token_ids != after.token_ids or abs(before.log_probs[0] - after.log_probs[0]) > 1e-4

        checkpoint_path = str(tmp_path / "checkpoint")
        actor_wg.save_checkpoint(checkpoint_path, global_step=1)
        actor_wg.optimize(training_ids)
        actor_wg.load_checkpoint(checkpoint_path)
        await manager.update_weights(global_steps=2)
        restored = await generate("restored")
        assert restored.extra_fields["global_steps"] == 2
        assert restored.token_ids == after.token_ids
        torch.testing.assert_close(torch.tensor(restored.log_probs), torch.tensor(after.log_probs), atol=1e-3, rtol=0)
        assert importlib.util.find_spec("vllm") is None
    finally:
        if replica is not None:
            for actor in replica.servers + replica.workers:
                ray.kill(actor, no_restart=True)
        if actor_wg is not None:
            for actor in actor_wg.workers:
                ray.kill(actor, no_restart=True)
        ray.shutdown()
        ray.cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
