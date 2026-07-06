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
Test vLLM rollout generation determinism.

Same-instance determinism is trivial (same RNG state). Cross-instance
determinism requires PYTHONHASHSEED set before Python starts, because
hash() and dict ordering are frozen at interpreter startup.

Tests:
1. Same-instance: rollout twice on one vLLM server (baseline).
2. Cross-instance vLLM: two separate server instances produce identical logprobs.
3. Cross-instance AgentLoop: two full AgentLoopManager pipelines produce identical logprobs.

Cross-instance RewardLoop determinism is verified by the separate E2E script:
  tests/experimental/reward_loop/run_determinism_e2e_with_rm.py

Environment overrides:
  VLLM_DETERMINISM_DENSE_MODEL_PATH  - policy model (default Qwen2.5-0.5B-Instruct)
  VLLM_DETERMINISM_N_GPUS            - GPUs for AgentLoop tests (default 1)
"""

import asyncio
import math
import os

import numpy as np
import ray
import torch
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.protocol import DataProto
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import get_rollout_replica_class

SEED = 42
DENSE_MODEL_PATH = os.path.expanduser(
    os.getenv("VLLM_DETERMINISM_DENSE_MODEL_PATH", "~/models/Qwen/Qwen2.5-0.5B-Instruct")
)

PROMPTS = [
    "Write one short sentence about deterministic generation.",
    "Explain what reproducibility means in one sentence.",
    "Describe batch invariance in one short phrase.",
]

RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_USE_V1": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "PYTHONHASHSEED": str(SEED),
    }
}


def _get_config_dir():
    config_dir = os.path.abspath("verl/verl/trainer/config")
    if not os.path.exists(config_dir):
        config_dir = os.path.abspath("verl/trainer/config")
    return config_dir


def _pearson_correlation(first, second):
    f = torch.tensor(first, dtype=torch.float64)
    s = torch.tensor(second, dtype=torch.float64)
    return torch.corrcoef(torch.stack([f, s], dim=0))[0, 1].item()


def _make_rollout_config(model_path, seed):
    with initialize_config_dir(config_dir=_get_config_dir(), version_base=None):
        config = compose(config_name="ppo_trainer")
    config.trainer.n_gpus_per_node = 1
    config.trainer.nnodes = 1
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.model.trust_remote_code = True
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    config.actor_rollout_ref.rollout.prompt_length = 128
    config.actor_rollout_ref.rollout.response_length = 256
    config.actor_rollout_ref.rollout.max_model_len = 512
    config.actor_rollout_ref.rollout.max_num_seqs = 8
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.4
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.full_determinism = True
    config.actor_rollout_ref.rollout.seed = seed
    config.actor_rollout_ref.rollout.scheduling_policy = "priority"
    return config


def _make_e2e_config(model_path, seed, n_gpus=2):
    """Config for full AgentLoopManager pipeline."""
    with initialize_config_dir(config_dir=_get_config_dir(), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                "actor_rollout_ref.actor.fsdp_config.param_offload=true",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
                "reward.reward_manager.name=dapo",
                "+reward.reward_kwargs.overlong_buffer_cfg.enable=False",
                "+reward.reward_kwargs.overlong_buffer_cfg.len=3072",
                "+reward.reward_kwargs.max_resp_len=4096",
            ],
        )
    config.trainer.n_gpus_per_node = n_gpus
    config.trainer.nnodes = 1
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.model.trust_remote_code = True
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    config.actor_rollout_ref.rollout.prompt_length = 128
    config.actor_rollout_ref.rollout.response_length = 256
    config.actor_rollout_ref.rollout.max_model_len = 512
    config.actor_rollout_ref.rollout.max_num_seqs = 8
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.4
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.full_determinism = True
    config.actor_rollout_ref.rollout.seed = seed
    config.actor_rollout_ref.rollout.scheduling_policy = "priority"
    config.actor_rollout_ref.rollout.calculate_log_probs = True
    config.actor_rollout_ref.rollout.temperature = 0.7
    config.actor_rollout_ref.rollout.top_p = 1.0
    config.actor_rollout_ref.rollout.n = 1
    config.actor_rollout_ref.rollout.agent.num_workers = 1
    config.actor_rollout_ref.rollout.skip_tokenizer_init = False
    config.actor_rollout_ref.rollout.disable_log_stats = True
    return config


# ──── Helpers for standalone vLLM tests ────


def _run_once(server, tokenizer, prompt_texts):
    refs = []
    for i, text in enumerate(prompt_texts):
        prompt_ids = normalize_token_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        refs.append(
            server._server_handle.generate.remote(
                request_id=f"determinism_{i}",
                prompt_ids=prompt_ids,
                sampling_params={"temperature": 0.7, "top_p": 1.0, "logprobs": True},
                priority=i,
                image_data=None,
            )
        )
    outputs = ray.get(refs, timeout=120.0)
    return [(o.token_ids, o.log_probs) for o in outputs]


def _create_vllm_server(config, replica_rank=0, is_reward_model=False, name_suffix=""):
    rollout_server_class = get_rollout_replica_class("vllm")
    server = rollout_server_class(
        replica_rank=replica_rank,
        config=config.actor_rollout_ref.rollout,
        model_config=config.actor_rollout_ref.model,
        gpus_per_node=1,
        is_reward_model=is_reward_model,
        name_suffix=name_suffix,
    )
    asyncio.run(server.init_standalone())
    return server


def _compare_logprobs(first, second, label):
    for i in range(len(first)):
        ids1, lp1 = first[i]
        ids2, lp2 = second[i]
        f1 = torch.tensor(lp1, dtype=torch.float32)
        f2 = torch.tensor(lp2, dtype=torch.float32)
        ids_match = ids1 == ids2
        lp_match = torch.equal(f1, f2)
        max_diff = torch.max(torch.abs(f1 - f2)).item() if not lp_match else 0.0
        corr = _pearson_correlation(lp1, lp2)
        print(
            f"  {label} prompt-{i}: ids_match={ids_match}, logprobs_match={lp_match}, "
            f"max_diff={max_diff:.6e}, corr={corr:.12f}"
        )
        assert ids_match
        assert lp_match, f"max_diff={max_diff}"
        assert math.isclose(corr, 1.0, abs_tol=1e-12), f"corr={corr}"


# ──── Helpers for AgentLoop tests ────


def _make_agent_loop_batch():
    return DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([[{"role": "user", "content": p}] for p in PROMPTS], dtype=object),
            "agent_name": np.array(["single_turn_agent"] * len(PROMPTS)),
            "data_source": np.array(["openai/gsm8k"] * len(PROMPTS)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(PROMPTS)),
        },
    )


def _compare_dataproto_logprobs(first, second, label):
    """Compare rollout_log_probs and responses at valid positions."""
    lp1 = first.batch["rollout_log_probs"]
    lp2 = second.batch["rollout_log_probs"]
    mask1 = first.batch["response_mask"].bool()
    mask2 = second.batch["response_mask"].bool()
    assert torch.equal(mask1, mask2), f"{label}: response_mask mismatch"
    assert torch.equal(first.batch["responses"][mask1], second.batch["responses"][mask2]), (
        f"{label}: token IDs mismatch"
    )
    v1 = lp1[mask1]
    v2 = lp2[mask2]
    assert torch.equal(v1, v2), f"{label}: logprob mismatch, max_diff={torch.max(torch.abs(v1 - v2)).item()}"
    corr = _pearson_correlation(v1.float().tolist(), v2.float().tolist())
    assert math.isclose(corr, 1.0, abs_tol=1e-12), f"{label}: corr={corr}"
    print(f"  {label}: logprobs_match=True, corr={corr:.12f}")


# ──── Test 1: Same-instance vLLM ────


def test_rollout_determinism_same_instance():
    ray.shutdown()
    ray.init(runtime_env=RAY_RUNTIME_ENV, ignore_reinit_error=True)

    config = _make_rollout_config(DENSE_MODEL_PATH, SEED)
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL_PATH, trust_remote_code=True)
    server = _create_vllm_server(config)

    r1 = _run_once(server, tokenizer, PROMPTS)
    r2 = _run_once(server, tokenizer, PROMPTS)
    _compare_logprobs(r1, r2, "[same-instance]")
    print("✓ Same-instance determinism verified")
    ray.shutdown()


# ──── Test 2: Cross-instance vLLM ────


def test_rollout_determinism_cross_instance():
    ray.shutdown()
    ray.init(runtime_env=RAY_RUNTIME_ENV, ignore_reinit_error=True)

    config = _make_rollout_config(DENSE_MODEL_PATH, SEED)
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL_PATH, trust_remote_code=True)

    s1 = _create_vllm_server(config, replica_rank=0, name_suffix="run1")
    r1 = _run_once(s1, tokenizer, PROMPTS)

    s2 = _create_vllm_server(config, replica_rank=0, name_suffix="run2")
    r2 = _run_once(s2, tokenizer, PROMPTS)

    _compare_logprobs(r1, r2, "[cross-instance]")
    print("✓ Cross-instance determinism verified")
    ray.shutdown()


# ──── Test 3: Cross-instance AgentLoop ────


def test_agent_loop_determinism_cross_instance():
    """Two full AgentLoopManager pipelines produce identical logprobs."""
    n_gpus = int(os.getenv("VLLM_DETERMINISM_N_GPUS", "1"))

    config = _make_e2e_config(DENSE_MODEL_PATH, SEED, n_gpus=n_gpus)
    batch = _make_agent_loop_batch()

    # Run 1
    ray.shutdown()
    ray.init(runtime_env=RAY_RUNTIME_ENV, ignore_reinit_error=True)
    mgr1 = init_agent_loop_manager(config)
    r1 = mgr1.generate_sequences(prompts=batch)
    ray.shutdown()

    # Run 2
    ray.init(runtime_env=RAY_RUNTIME_ENV, ignore_reinit_error=True)
    mgr2 = init_agent_loop_manager(config)
    r2 = mgr2.generate_sequences(prompts=batch)
    ray.shutdown()

    _compare_dataproto_logprobs(r1, r2, "[agent-loop-cross-instance]")
    print("✓ AgentLoop cross-instance determinism verified")


if __name__ == "__main__":
    test_rollout_determinism_same_instance()
    test_rollout_determinism_cross_instance()
    test_agent_loop_determinism_cross_instance()
