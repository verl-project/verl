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
"""Rollout with vLLM / SGLang, recompute old_log_probs with Megatron, and report the rollout-vs-actor log-prob gap.

Exits non-zero if k3_kl exceeds --max-k3-kl. Trailing arguments are passed to Hydra as config overrides.

Examples (8 GPUs):
    # bf16 rollout
    .venv/bin/python scripts/rollout_logprob_diff.py --model-path ~/models/Qwen3.5-0.8B
    # fp8 blockwise rollout on a MoE model
    .venv/bin/python scripts/rollout_logprob_diff.py --model-path ~/models/Qwen3.5-35B-A3B --ep 8 \\
        --quant-linear FP8_BLOCK --quant-moe FP8_BLOCK
    # MXFP8 rollout on Blackwell (linear only for a dense model, linear + MoE for a MoE model)
    .venv-vllm/bin/python scripts/rollout_logprob_diff.py --model-path ~/models/Qwen3.5-0.8B \\
        --quant-linear MXFP8 --quant-ignore 're:.*linear_attn\\.in_proj_[ab]$'
    .venv-vllm/bin/python scripts/rollout_logprob_diff.py --model-path ~/models/Qwen3.5-35B-A3B --ep 8 \\
        --quant-linear MXFP8 --quant-moe MXFP8 --qat-mode mxfp8 \\
        --quant-ignore 're:.*linear_attn\\.in_proj_[ab]$' \\
        --qat-ignore '*linear_attn.in_proj_a' --qat-ignore '*linear_attn.in_proj_b' \\
        --r3-router-replay
    # MXFP4 expert weights with dynamic MXFP8 expert activations
    .venv-vllm/bin/python scripts/rollout_logprob_diff.py --model-path ~/models/Qwen3.5-35B-A3B --ep 8 \\
        --quant-moe MXFP4 --qat-mode mxfp4 --r3-router-replay
    # NVFP4 W4A4 experts with FlashInfer CUTLASS
    .venv-vllm/bin/python scripts/rollout_logprob_diff.py --model-path ~/models/Qwen3.5-35B-A3B --ep 8 \\
        --quant-moe NVFP4 --qat-mode nvfp4
    # same, with Megatron ModelOpt QAT fake-quantizing to the rollout scheme
    .venv/bin/python scripts/rollout_logprob_diff.py --model-path ~/models/Qwen3.5-35B-A3B --ep 8 \\
        --quant-linear FP8_BLOCK --quant-moe FP8_BLOCK --qat-mode fp8
    # SGLang rollout
    .venv-sglang/bin/python scripts/rollout_logprob_diff.py --engine sglang --model-path ~/models/Qwen3.5-35B-A3B \\
        --ep 8 --quant-linear FP8_BLOCK --quant-moe FP8_BLOCK
"""

import argparse
import json
import os
import sys

import numpy as np
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

import verl
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics
from verl.utils import omega_conf_to_dataclass
from verl.utils import tensordict_utils as tu
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.device import get_device_name
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

QUESTIONS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. "
    "Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. "
    "How much more money does Betty need to buy the wallet?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Write a short poem about the ocean.",
    "Explain the difference between TCP and UDP in a few sentences.",
    "Let's play a role playing game. Your name is Alice, your favorite color is blue. Introduce yourself.",
    "What is the derivative of x^3 * sin(x)?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="~/models/Qwen3.5-35B-A3B")
    parser.add_argument("--engine", default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--tp", type=int, default=2, help="Megatron tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Megatron pipeline parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Megatron context parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Megatron expert parallel size")
    parser.add_argument("--etp", type=int, default=1, help="Megatron expert tensor parallel size")
    parser.add_argument("--gen-tp", type=int, default=2, help="rollout tensor parallel size")
    parser.add_argument("--quant-linear", default=None, help="rollout.quantization.linear preset: FP8_BLOCK or MXFP8")
    parser.add_argument(
        "--quant-moe",
        default=None,
        help="rollout.quantization.moe preset: FP8_BLOCK, MXFP8, MXFP4, or NVFP4",
    )
    parser.add_argument(
        "--quant-ignore",
        action="append",
        default=[],
        help='extra rollout ignore pattern; repeat as needed, e.g. "re:.*linear_attn\\\\.in_proj_[ab]$"',
    )
    parser.add_argument("--qat-mode", default=None, help="Megatron ModelOpt QAT mode matching the rollout")
    parser.add_argument(
        "--qat-ignore",
        action="append",
        default=[],
        help="extra Megatron ModelOpt fnmatch ignore pattern; repeat as needed",
    )
    parser.add_argument(
        "--r3-router-replay",
        action="store_true",
        help="replay rollout expert choices in the Megatron log-prob recomputation",
    )
    parser.add_argument(
        "--refit-rounds",
        type=int,
        default=1,
        help="generate and measure after this many consecutive live-engine weight refits",
    )
    parser.add_argument("--max-k3-kl", type=float, default=1e-2)
    parser.add_argument("overrides", nargs="*", help="extra Hydra overrides, e.g. actor_rollout_ref.rollout.n=8")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DictConfig:
    config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "model_engine=megatron",
                f"trainer.n_gpus_per_node={args.n_gpus}",
                "trainer.nnodes=1",
                "actor_rollout_ref.actor.optim.total_training_steps=1",
                "actor_rollout_ref.actor.use_dynamic_bsz=True",
                "actor_rollout_ref.actor.megatron.use_remove_padding=True",
                f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={args.tp}",
                f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={args.pp}",
                f"actor_rollout_ref.actor.megatron.context_parallel_size={args.cp}",
                f"actor_rollout_ref.actor.megatron.expert_model_parallel_size={args.ep}",
                f"actor_rollout_ref.actor.megatron.expert_tensor_parallel_size={args.etp}",
                "actor_rollout_ref.actor.megatron.param_offload=True",
                "actor_rollout_ref.actor.megatron.optimizer_offload=True",
                "++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto",
                "+actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True",
                "+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True",
                f"actor_rollout_ref.rollout.name={args.engine}",
                "actor_rollout_ref.rollout.mode=async",
                f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.gen_tp}",
                "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
                "actor_rollout_ref.rollout.prompt_length=1024",
                "actor_rollout_ref.rollout.response_length=2048",
                "actor_rollout_ref.rollout.n=4",
                "actor_rollout_ref.rollout.agent.num_workers=2",
                "actor_rollout_ref.rollout.skip_tokenizer_init=True",
                *args.overrides,
            ],
        )

    config.actor_rollout_ref.model.path = os.path.expanduser(args.model_path)
    config.actor_rollout_ref.rollout.quantization.linear = args.quant_linear
    config.actor_rollout_ref.rollout.quantization.moe = args.quant_moe
    config.actor_rollout_ref.rollout.quantization.ignore = args.quant_ignore
    if args.qat_mode:
        config.actor_rollout_ref.actor.megatron.qat.enable = True
        config.actor_rollout_ref.actor.megatron.qat.mode = args.qat_mode
        config.actor_rollout_ref.actor.megatron.qat.ignore_patterns = [
            *config.actor_rollout_ref.actor.megatron.qat.ignore_patterns,
            *args.qat_ignore,
        ]
    if args.r3_router_replay:
        config.actor_rollout_ref.actor.megatron.router_replay.mode = "R3"
        config.actor_rollout_ref.rollout.enable_rollout_routing_replay = True
    return config


def init_workers(config: DictConfig) -> tuple[AgentLoopManager, RayWorkerGroup, CheckpointEngineManager]:
    """Hybrid actor/rollout workers + agent loop, without reward model."""
    pool_id = "global_pool"
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping={Role.ActorRollout: pool_id},
    )
    resource_pool_manager.create_resource_pool()
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)

    class_dict = {
        "actor_rollout": RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker), config=config.actor_rollout_ref, role="actor_rollout"
        )
    }
    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=create_colocated_worker_cls(class_dict=class_dict),
        device_name=get_device_name(),
    )
    actor_rollout_wg = wg_dict.spawn(prefix_set=class_dict.keys())["actor_rollout"]
    actor_rollout_wg.init_model()

    reward_loop_manager = RewardLoopManager(config=config, rm_resource_pool=None)
    llm_server_manager = LLMServerManager.create(config=config, worker_group=actor_rollout_wg)
    agent_loop_manager = AgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
        reward_loop_worker_handles=reward_loop_manager.reward_loop_workers,
    )
    checkpoint_manager = CheckpointEngineManager(
        config=omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine),
        actor_wg=actor_rollout_wg,
        replicas=llm_server_manager.get_replicas(),
    )
    checkpoint_manager.sleep_replicas()
    checkpoint_manager.update_weights()
    return agent_loop_manager, actor_rollout_wg, checkpoint_manager


def compute_old_log_prob(actor_rollout_wg: RayWorkerGroup, batch: DataProto) -> DataProto:
    """Same as RayPPOTrainer._compute_old_log_prob."""
    batch_td = left_right_2_no_padding(batch.to_tensordict())
    tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
    output = actor_rollout_wg.compute_log_prob(batch_td)
    log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)
    return DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": log_probs.float()}))


def main() -> int:
    args = parse_args()
    if args.refit_rounds < 1:
        raise ValueError("--refit-rounds must be at least 1")
    config = build_config(args)

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            }
        }
    )
    agent_loop_manager, actor_rollout_wg, checkpoint_manager = init_workers(config)

    raw_prompts = [[{"role": "user", "content": q}] for q in QUESTIONS]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "agent_name": np.array(["single_turn_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
        },
    ).repeat(config.actor_rollout_ref.rollout.n)

    refit_runs = []
    for round_index in range(args.refit_rounds):
        if round_index:
            checkpoint_manager.update_weights()
        result = agent_loop_manager.generate_sequences(prompts=batch)
        checkpoint_manager.sleep_replicas()
        result.meta_info.pop("timing", None)

        result.meta_info["temperature"] = config.actor_rollout_ref.rollout.temperature
        result.meta_info["global_token_num"] = torch.sum(result.batch["attention_mask"], dim=-1).tolist()
        result = result.union(compute_old_log_prob(actor_rollout_wg, result))

        old_log_probs = result.batch["old_log_probs"]
        rollout_log_probs = result.batch["rollout_log_probs"]
        response_mask = result.batch["response_mask"].bool()

        metrics = compute_offpolicy_metrics(
            old_log_prob=old_log_probs, rollout_log_prob=rollout_log_probs, response_mask=response_mask
        )
        metrics.update(calculate_debug_metrics(result))
        log_prob_abs_diff = (old_log_probs - rollout_log_probs).abs()[response_mask]
        metrics["log_prob_abs_diff_mean"] = log_prob_abs_diff.mean().item()
        metrics["log_prob_abs_diff_max"] = log_prob_abs_diff.max().item()
        metrics["num_response_tokens"] = int(response_mask.sum().item())
        refit_runs.append(metrics)

    metrics = dict(refit_runs[-1])
    if len(refit_runs) > 1:
        metrics["refit_runs"] = refit_runs
    print(json.dumps(metrics, indent=2))
    ray.shutdown()

    k3_kl = metrics["k3_kl"]
    if not (np.isfinite(metrics["kl"]) and np.isfinite(k3_kl)) or k3_kl >= args.max_k3_kl:
        print(f"FAILED: k3_kl {k3_kl} (threshold {args.max_k3_kl})")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
