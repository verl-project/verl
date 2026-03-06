# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from omegaconf import OmegaConf

from verl.workers.fsdp_workers import ActorRolloutRefWorker


def test_actor_rollout_ref_worker_actor_ref_model():
    """Test specifying different reference/actor model"""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8888"

    actor_model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    ref_model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")
    config_str = f"""
    model:
      path: {actor_model_path}
    actor:
      _target_: verl.workers.config.FSDPActorConfig
      strategy: fsdp
      fsdp_config:
        _target_: verl.workers.config.FSDPEngineConfig
        fsdp_size: -1
        forward_prefetch: false
      profiler:
        tool: torch_memory
        save_path: ./mem_snapshots
        tool_config:
          torch_memory:
            _target_: verl.utils.profiler.config.TorchMemoryToolConfig
            trace_alloc_max_entries: 100000
            stack_depth: 32
    ref:
      model:
        path: {ref_model_path}
      fsdp_config:
        _target_: verl.workers.config.FSDPEngineConfig
        fsdp_size: -1
      profiler:
        tool: torch_memory
        save_path: ./mem_snapshots
        tool_config:
          torch_memory:
            _target_: verl.utils.profiler.config.TorchMemoryToolConfig
            trace_alloc_max_entries: 100000
            stack_depth: 32
      log_prob_micro_batch_size: 1
      ulysses_sequence_parallel_size: 1
      entropy_from_logits_with_chunking: false
    """
    dict_conf = OmegaConf.create(config_str)
    actor_rollout_ref_worker = ActorRolloutRefWorker(dict_conf, role="ref")
    actor_rollout_ref_worker.init_model()

    model_config = actor_rollout_ref_worker.ref_module_fsdp._fsdp_wrapped_module.config
    assert model_config.hidden_size == 1536

    # set ref.model to null, fallback to default case where actor is the same as reference
    dict_conf["ref"]["model"] = None
    actor_rollout_ref_worker = ActorRolloutRefWorker(dict_conf, role="ref")
    actor_rollout_ref_worker.init_model()

    model_config = actor_rollout_ref_worker.ref_module_fsdp._fsdp_wrapped_module.config
    assert model_config.hidden_size == 896
