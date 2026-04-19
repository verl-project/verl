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
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

from ..model import MtpConfig
from ..rollout import AgentLoopConfig, CheckpointEngineConfig, MultiTurnConfig, PrometheusConfig, SamplingConfig

__all__ = [
    "DiffusionRolloutAlgoConfig",
    "DiffusionSamplingConfig",
    "DiffusionRolloutConfig",
]


@dataclass
class DiffusionRolloutAlgoConfig(BaseConfig):
    # for SDE-based diffusion process
    noise_level: float = 1.0
    sde_type: str = "sde"
    sde_window_size: Optional[int] = None
    sde_window_range: list[int] = field(default_factory=lambda: [0, 5])


@dataclass
class DiffusionSamplingConfig(SamplingConfig):
    num_inference_steps: int = 40
    seed: int = 42
    algo: DiffusionRolloutAlgoConfig = field(default_factory=DiffusionRolloutAlgoConfig)


@dataclass
class DiffusionRolloutConfig(BaseConfig):
    _mutable_fields = {"max_model_len", "load_format", "engine_kwargs", "prompt_length", "expert_parallel_size"}

    name: Optional[str] = MISSING
    mode: str = "async"
    nnodes: int = 0
    n_gpus_per_node: int = 8
    n: int = 1

    prompt_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    enforce_eager: bool = False
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    logprobs_mode: Optional[str] = "processed_logprobs"
    scheduling_policy: Optional[str] = "fcfs"

    val_kwargs: DiffusionSamplingConfig = field(default_factory=DiffusionSamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size_per_gpu: Optional[int] = None

    disable_log_stats: bool = True

    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Use Prometheus to collect and monitor rollout statistics
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)

    # Checkpoint Engine config for update weights from trainer to rollout
    checkpoint_engine: CheckpointEngineConfig = field(default_factory=CheckpointEngineConfig)

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    load_format: str = "dummy"

    layered_summon: bool = False

    skip_tokenizer_init: bool = True

    quantization: Optional[str] = None

    enable_rollout_routing_replay: bool = False

    enable_sleep_mode: bool = True

    mtp: Optional[MtpConfig] = field(default_factory=MtpConfig)

    height: int = 512

    width: int = 512

    num_inference_steps: int = 10

    true_cfg_scale: float = 4.0

    max_sequence_length: int = 512

    guidance_scale: Optional[float] = None

    profiler: Optional[ProfilerConfig] = None

    algo: Optional[DiffusionRolloutAlgoConfig] = field(default_factory=DiffusionRolloutAlgoConfig)

    external_lib: Optional[str] = None

    def __post_init__(self):
        """Validate the diffusion rollout config"""
        if self.mode == "sync":
            raise ValueError(
                "Rollout mode 'sync' has been removed. Please set "
                "`actor_rollout_ref.rollout.mode=async` or remove the mode setting entirely."
            )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm_omni":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )
