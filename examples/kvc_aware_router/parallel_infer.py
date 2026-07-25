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

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import ray
from datasets import load_dataset
from omegaconf import DictConfig

import verl
from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.workers.rollout.llm_server import LLMServerManager

# Setup basic logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=os.getenv("VERL_LOGGING_LEVEL", "INFO")
)
logger = logging.getLogger(__name__)

# Ray's default idle-worker reaper (~10 s) kills agent workers between
# dispatch gaps, ending the job prematurely.  Use a very large threshold
# so long-running agent loops are not interrupted.
_RAY_IDLE_WORKER_TIMEOUT_MS = int(os.getenv("RAY_IDLE_WORKER_TIMEOUT_MS", str(2**30 - 1)))


def init_config(args: argparse.Namespace) -> DictConfig:
    """Initialize the configuration from hydra and override with command-line arguments."""
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    # This example always uses the KV-cache-aware router.
    overrides = [
        "rollout/router@actor_rollout_ref.rollout.router_config=kvcaware",
        "actor_rollout_ref.rollout.router_strategy=kvcaware",
    ]
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)

    rollout = config.actor_rollout_ref.rollout

    # Rollout overrides (prompt_length / response_length / n_gpus_per_node are
    # oc.select-derived from the data / trainer nodes set below, so not repeated).
    rollout.agent.agent_loop_config_path = os.path.expanduser(args.agent_config_path)
    rollout.agent.num_workers = args.num_workers
    rollout.multi_turn.max_assistant_turns = args.max_turns
    rollout.temperature = args.temperature
    rollout.top_p = args.top_p
    rollout.nnodes = args.nnodes
    rollout.name = args.engine
    rollout.n = args.n
    rollout.tensor_model_parallel_size = args.tensor_parallel_size
    rollout.gpu_memory_utilization = 0.8
    rollout.max_num_seqs = args.max_num_seqs
    if args.max_model_len is not None:
        rollout.max_model_len = args.max_model_len
    rollout.disable_log_stats = False  # expose engine metrics on /metrics for the kvcaware collector

    # Hardware (rollout.n_gpus_per_node derives from trainer via oc.select)
    config.trainer.nnodes = args.nnodes
    config.trainer.n_gpus_per_node = args.n_gpus_per_node

    config.actor_rollout_ref.model.path = os.path.expanduser(args.model_path)

    # Data (rollout.prompt_length / response_length derive from these via oc.select)
    config.data.max_prompt_length = args.prompt_length
    config.data.max_response_length = args.response_length

    # vLLM engine kwargs: MFU metric (always on) + optional mooncake connector / kv-events.
    vllm_kwargs: dict = {"enable_mfu_metrics": True}
    if args.enable_mooncake:
        # Cross-replica KV sharing via mooncake (config via MOONCAKE_CONFIG_PATH env, not
        # extra_config). The connector class differs by backend: GPU build uses
        # "MooncakeStoreConnector"; vllm-ascend uses "MooncakeConnectorStoreV1".
        mooncake_connector = "MooncakeConnectorStoreV1" if args.device == "ascend" else "MooncakeStoreConnector"
        vllm_kwargs["kv_transfer_config"] = {
            "kv_connector": mooncake_connector,
            "kv_role": "kv_both",
            "kv_connector_extra_config": {},
        }
    if args.kv_events:
        # vLLM kv-events (zmq publisher) — KVCAware router load signal (retained-cache
        # occupancy). Endpoint ports are placeholders (verl assigns ephemeral).
        vllm_kwargs["kv-events-config"] = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": "tcp://*:0",
            "replay_endpoint": "tcp://*:0",
        }
    rollout.engine_kwargs = {"vllm": vllm_kwargs}

    # Optional KVCAware router strategy[0] overrides — each falls back to the
    # kvcaware.yaml value when the flag is omitted. The override above composes
    # router_config, so strategies is a non-empty list.
    strat0 = rollout.router_config.strategies[0]
    if args.alpha is not None:
        strat0.alpha = args.alpha
    if args.load_threshold is not None:
        strat0.load_threshold = args.load_threshold
    if args.slow_cut is not None:
        strat0.slow_cut = args.slow_cut
    if args.overload_mode is not None:
        strat0.overload_mode = args.overload_mode

    return config


def run_inference(args: argparse.Namespace):
    """Run the inference pipeline using the provided arguments."""
    # vLLM's mooncake connector reads MOONCAKE_CONFIG_PATH (not extra_config).
    # Set before ray.init so Ray-spawned workers inherit it.
    if args.enable_mooncake and args.mooncake_config_path:
        os.environ["MOONCAKE_CONFIG_PATH"] = os.path.expanduser(args.mooncake_config_path)

    # 1. Init Ray — disable idle-worker reaper so agent workers survive
    # dispatch gaps (default ~10 s threshold would kill them prematurely).
    ray.init(_system_config={"idle_worker_killing_time_threshold_ms": _RAY_IDLE_WORKER_TIMEOUT_MS})

    # 2. Init rollout manager
    logger.info("Initializing configuration and AgentLoopManager...")
    config = init_config(args)
    llm_server_manager = LLMServerManager.create(config=config)
    agent_loop_manager = AgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
    )

    # 3. Load dataset
    data_path = os.path.expanduser(args.data_path)
    logger.info(f"Loading dataset from: {data_path}")
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    if args.shuffle:
        logger.info("Shuffling dataset (seed=%d) before sampling", args.seed)
        dataset = dataset.shuffle(seed=args.seed)
    # Slice on the lazy dataset before materializing, so the full parquet is
    # never converted to Python objects when --max-samples is small.
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        logger.info("Using first %d samples (--max-samples=%d)", len(dataset), args.max_samples)
    samples = dataset.to_list()

    # 4. Prepare batch data
    logger.info("Preparing data batch...")
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([sample["prompt"] for sample in samples], dtype=object),
            "agent_name": np.array([sample["agent_name"] for sample in samples], dtype=object),
            "tools_kwargs": np.array([sample["extra_info"]["tools_kwargs"] for sample in samples], dtype=object),
        },
        meta_info={"validate": True},
    ).repeat(config.actor_rollout_ref.rollout.n)

    # 5. Generate sequences
    logger.info("Starting sequence generation...")
    size_divisor = config.actor_rollout_ref.rollout.agent.num_workers
    batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
    output_padded = agent_loop_manager.generate_sequences(batch_padded)
    output = unpad_dataproto(output_padded, pad_size=pad_size)

    # 6. Process results
    rm_scores = output.batch["rm_scores"].sum(dim=-1).tolist()
    mean_score = float(np.mean(rm_scores)) if len(rm_scores) > 0 else 0.0

    logger.info(f"Generation completed. Mean RM Score: {mean_score:.4f}")
    print(f"\n=> Mean RM Score: {mean_score:.4f}\n")

    # 7. Optionally persist a machine-readable result file (used by eval_checkpoints.py).
    if args.result_path:
        result_path = os.path.expanduser(args.result_path)
        os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        result = {
            "model_path": os.path.expanduser(args.model_path),
            "data_path": data_path,
            "agent_config_path": os.path.expanduser(args.agent_config_path),
            "n": config.actor_rollout_ref.rollout.n,
            "num_samples": len(rm_scores),
            "mean_rm_score": mean_score,
            "rm_scores": rm_scores,
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote result file to: {result_path}")

    return mean_score


def main():
    parser = argparse.ArgumentParser(description="Uni-Agent Inference Runner")

    # Input / Output configs
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the input dataset (Parquet format).",
    )
    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        default=None,
        help="Path to the local model checkpoint.",
    )
    parser.add_argument(
        "--agent-config-path",
        type=str,
        default=None,
        help="Path to the agent loop configuration YAML.",
    )
    parser.add_argument(
        "--result-path",
        type=str,
        default=None,
        help="Optional path to write a JSON result file (mean reward and per-rollout scores).",
    )
    # Inference parameters
    parser.add_argument("--max-turns", type=int, default=100, help="Maximum number of interaction turns per episode.")
    parser.add_argument("--prompt-length", type=int, default=4096, help="Maximum prompt length (tokens).")
    parser.add_argument("--response-length", type=int, default=8192, help="Maximum response length (tokens).")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Sampling top-p (nucleus sampling).")
    parser.add_argument("--n", type=int, default=1, help="Number of rollouts per prompt (N).")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Max number of samples to run. Use -1 for no limit (full dataset).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before slicing (--max-samples / --n). Aligns with fully_async data.shuffle.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --shuffle.")

    # Execution / Engine configs
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=["vllm", "sglang"],
        help="Inference engine backend (e.g., vllm or sglang).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "ascend"],
        help="Target backend: 'gpu' or 'ascend'",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Number of agent rollout workers.")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes to run the job.")
    parser.add_argument("--n-gpus-per-node", type=int, default=2, help="Number of GPUs per node.")
    parser.add_argument(
        "--tensor-parallel-size", "--tp", type=int, default=1, help="Tensor parallel size for the model."
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length (tokens). If unset the engine default is used.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of concurrent sequences per engine.",
    )
    parser.add_argument(
        "--enable-mooncake",
        action="store_true",
        help="Attach MooncakeStoreConnector for cross-replica KV sharing (a mooncake master must run separately).",
    )
    parser.add_argument(
        "--mooncake-config-path",
        type=str,
        default="mooncake_config.json",
        help="Path to the mooncake config JSON (used with --enable-mooncake).",
    )
    parser.add_argument(
        "--kv-events",
        action="store_true",
        help="Enable vLLM kv-events zmq publisher for retained-cache occupancy collection. "
        "Required for KVCAware router load signal and standalone collector metrics.",
    )

    # KVCAware router strategy[0] overrides (each falls back to kvcaware.yaml when omitted).
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="KVCAware strategy[0] α (cache vs load blend, [0,1]). Overrides kvcaware.yaml when set.",
    )
    parser.add_argument(
        "--load-threshold",
        type=float,
        default=None,
        help="KVCAware strategy[0] load_threshold (overload when load > threshold, (0,1)). "
        "Overrides kvcaware.yaml when set.",
    )
    parser.add_argument(
        "--slow-cut",
        type=str,
        choices=["prefix-load-aware", "least-inflight", "capacity-token-aware"],
        default=None,
        help="KVCAware strategy[0] slow_cut fallback scoring mode. Overrides kvcaware.yaml when set.",
    )
    parser.add_argument(
        "--overload-mode",
        type=str,
        choices=["None", "kv_cache_usage_perc", "kv_load"],
        default=None,
        help="KVCAware strategy[0] overload_mode for the sticky short-circuit ",
    )

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
