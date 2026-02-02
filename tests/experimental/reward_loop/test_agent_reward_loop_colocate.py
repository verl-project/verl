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
import os

import ray
from hydra import compose, initialize_config_dir
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.main_ppo import create_rl_sampler
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.device import get_device_name
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker


def test_agent_loop_reward_manager():
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_trainer")

    rollout_model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    reward_model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")

    # actor_rollout_ref config
    config.data.return_raw_chat = True
    config.data.max_prompt_length = 1024
    config.data.max_response_length = 4096
    config.actor_rollout_ref.model.path = rollout_model_path
    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    config.actor_rollout_ref.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.8
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.skip_tokenizer_init = True
    config.trainer.n_gpus_per_node = 8
    config.trainer.nnodes = 1

    config.reward_model.reward_manager = "dapo"
    config.reward_model.enable = True
    config.reward_model.enable_resource_pool = False
    config.reward_model.n_gpus_per_node = 8
    config.reward_model.model.path = reward_model_path
    config.reward_model.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.reward_model.rollout.gpu_memory_utilization = 0.8
    config.reward_model.rollout.tensor_model_parallel_size = 2
    config.reward_model.rollout.skip_tokenizer_init = False
    config.reward_model.rollout.prompt_length = 5120
    config.reward_model.rollout.response_length = 4096
    config.custom_reward_function.path = "tests/experimental/reward_loop/reward_fn.py"
    config.custom_reward_function.name = "compute_score_gsm8k"

    # 1. init reward model manager
    actor_rollout_cls = (
        AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
    )
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=None)
    resource_pool_manager.create_resource_pool()
    resource_pool = resource_pool_manager.resource_pool_dict[global_pool_id]
    actor_rollout_cls = RayClassWithInitArgs(
        cls=ray.remote(actor_rollout_cls), config=config.actor_rollout_ref, role="actor_rollout"
    )
    actor_rollout_wg = RayWorkerGroup(
        resource_pool=resource_pool, ray_cls_with_init=actor_rollout_cls, device_name=get_device_name()
    )
    actor_rollout_wg.init_model()

    agent_loop_manager = AgentLoopManager(config, worker_group=actor_rollout_wg)
    # sleep rollout replicas
    checkpoint_manager = CheckpointEngineManager(
        backend=config.actor_rollout_ref.rollout.checkpoint_engine.backend,
        trainer=actor_rollout_wg,
        replicas=agent_loop_manager.rollout_replicas,
    )
    checkpoint_manager.sleep_replicas()
    reward_loop_manager = RewardLoopManager(config, rm_resource_pool=resource_pool)

    # 2. init test data
    local_folder = os.path.expanduser("~/data/gsm8k/")

    data_files = [os.path.join(local_folder, "train.parquet")]
    tokenizer = AutoTokenizer.from_pretrained(rollout_model_path)

    dataset = RLHFDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=None,
    )

    batch_size = 64
    sampler = create_rl_sampler(config.data, dataset)
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    # 3. generate responses
    batch_dict = next(iter(dataloader))
    batch = DataProto.from_single_dict(batch_dict)

    def _get_gen_batch(batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    # wake up rollout replicas via update_weight
    checkpoint_manager.update_weights()
    gen_batch = _get_gen_batch(batch)
    gen_batch = agent_loop_manager.generate_sequences(gen_batch)
    checkpoint_manager.sleep_replicas()

    batch = batch.union(gen_batch)
    rm_outputs = reward_loop_manager.compute_rm_score(batch)

    for output in rm_outputs[:5]:
        print(output.non_tensor_batch)

    print("done")

    ray.shutdown()
