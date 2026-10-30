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
import asyncio
import logging
import os

import numpy as np
import ray
import uvicorn
import yaml
from fastapi import FastAPI

from verl.utils.tokenizer import get_processor_token_id
from verl.workers.config.rollout import PrometheusConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


class _UvicornServerAutoPort(uvicorn.Server):
    """Uvicorn Server that reports the system-assigned port when port=0."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.actual_port: int | None = None
        self._startup_done: asyncio.Event = asyncio.Event()

    async def startup(self, sockets=None) -> None:
        try:
            await super().startup(sockets=sockets)
            if self.servers and self.config.port == 0:
                sock = self.servers[0].sockets[0]
                self.actual_port = sock.getsockname()[1]
            else:
                self.actual_port = self.config.port
        finally:
            self._startup_done.set()

    async def get_port(self) -> int | None:
        await self._startup_done.wait()
        return self.actual_port


async def run_uvicorn(app: FastAPI, server_args, server_address) -> tuple[int, asyncio.Task]:
    app.server_args = server_args
    config = uvicorn.Config(app, host=server_address, port=0, log_level="warning")
    server = _UvicornServerAutoPort(config)
    server_task = asyncio.create_task(server.serve())
    server_port = await server.get_port()
    if server_port is None:
        # server.startup() failed. await the task to re-raise exception from server.serve()
        await server_task

        # Fails on unexpected situation.
        raise RuntimeError("Unexpected: HTTP server started without reporting listened port")
    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


def qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Collapse expanded Qwen2-VL/GLM image placeholders before backend processing.

    vLLM expands each placeholder again using the supplied image/video data.
    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if (
        processor is not None
        and hasattr(processor, "image_processor")
        and any(
            name in processor.image_processor.__class__.__name__
            for name in ("Qwen2VLImageProcessor", "Glm5NextImageProcessor")
        )
    ):
        prompt_ids = np.array(prompt_ids)
        mask = np.ones(len(prompt_ids), dtype=bool)
        is_value = (prompt_ids == processor.image_token_id) | (prompt_ids == processor.video_token_id)
        mask[1:] &= ~(is_value[1:] & (prompt_ids[1:] == prompt_ids[:-1]))
        return prompt_ids[mask].tolist()
    else:
        return prompt_ids


def get_vision_placeholder_token_ids(processor) -> list[int]:
    """Vision placeholder token ids the policy must never sample.

    An `<|image_pad|>` or `<|video_pad|>` token only means something when a real image or video
    sits behind it: the k-th run of placeholders pairs with the k-th row of `image_grid_thw` /
    `video_grid_thw`. Nothing stops the policy from sampling one anyway -- it is an ordinary entry
    of the vocabulary -- and then that pairing silently breaks. Every downstream consumer trusts
    it, so each one has its own way of dying: `get_rope_index` walks off the end of the grid
    iterator, and `merge_multimodal_embeddings` scatters into a mask that no longer matches the
    embedding count. Both take the whole training job down with them.

    A tool that returns an image is unaffected: that image arrives in the next turn's prompt, not
    in what the model generates.

    Returns an empty list for text-only models, leaving sampling untouched.
    """
    token_ids = []
    for modality in ("image", "video"):
        token_id = get_processor_token_id(processor, modality)
        if token_id is not None:
            token_ids.append(token_id)
    return token_ids


def extract_response_topk_logprobs(logprobs, k: int) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Sampled-token log-probs and the sampler top-k head per generated token.

    ``logprobs`` is a vLLM ``FlatLogprobs`` (``SamplingParams(flat_logprobs=True)``), which stores
    ``[sampled, top-1, ..., top-k]`` for every position in flat lists. Reshaping them avoids
    rebuilding one Logprob dict per position on the server event loop, and numpy arrays avoid
    shipping millions of Python scalars through Ray.

    Returns:
        Tuple containing:
            sampled_log_probs: Sampled-token log-probs, length T.
            ids: Top-k token ids ordered by rank, shape (T, k), dtype int32.
            log_probs: Top-k log-probs matching ``ids``, shape (T, k), dtype float32.
    """
    if k <= 0:
        raise ValueError("the sampler head size must be positive")
    num_tokens, width = len(logprobs), k + 1
    if len(logprobs.token_ids) != num_tokens * width or len(logprobs.logprobs) != num_tokens * width:
        raise ValueError(f"expected {width} logprob entries per generated token, got {len(logprobs.token_ids)} total")
    # Checking only the total permits a short row and a long row to cancel out. Never invent
    # missing slots: an -inf slot proves a zero probability only when the provider returned it.
    if hasattr(logprobs, "start_indices") and hasattr(logprobs, "end_indices"):
        if logprobs.start_indices != list(range(0, num_tokens * width, width)) or logprobs.end_indices != list(
            range(width, (num_tokens + 1) * width, width)
        ):
            raise ValueError(f"expected {width} logprob entries per generated token with contiguous flat rows")
    ids = np.asarray(logprobs.token_ids, dtype=np.int32).reshape(num_tokens, width)
    log_probs = np.asarray(logprobs.logprobs, dtype=np.float32).reshape(num_tokens, width)
    head_ids, head_log_probs = ids[:, 1:], log_probs[:, 1:]
    if np.any(ids < 0) or np.any(np.isnan(log_probs)) or np.any(np.isposinf(log_probs)) or np.any(log_probs > 1e-6):
        raise ValueError("sampler top-k ids must be nonnegative and log-probs must be finite or -inf")
    if not np.all(np.isfinite(log_probs[:, 0])) or np.any(~np.isfinite(head_log_probs).any(-1)):
        raise ValueError("the sampled token and at least one head token must have finite log-probs")
    if np.any(head_log_probs[:, :-1] < head_log_probs[:, 1:]):
        raise ValueError("sampler top-k log-probs must be ordered by rank")
    sorted_ids = np.sort(head_ids, axis=-1)
    if np.any(sorted_ids[:, :-1] == sorted_ids[:, 1:]):
        raise ValueError("sampler top-k ids must be unique within each head")
    if np.any(np.exp(head_log_probs).sum(-1) > 1.0 + 1e-5):
        raise ValueError("sampler top-k head probability mass must not exceed one")
    rows, columns = np.nonzero(head_ids == ids[:, :1])
    if not np.allclose(head_log_probs[rows, columns], log_probs[rows, 0], atol=1e-6, rtol=1e-6):
        raise ValueError("sampled-token log-probs must match the corresponding sampler head entries")
    return log_probs[:, 0].tolist(), ids[:, 1:].copy(), log_probs[:, 1:].copy()


def validate_topk_logprob_request(sampling_params) -> None:
    """Require a global top-k request before using provider -inf as support evidence.

    vLLM's custom-token logprob path replaces the ranked head with requested ids and may pad
    missing records with -inf. Its FlatLogprobs rank labels alone cannot distinguish it.
    """
    if sampling_params.logprobs:
        if getattr(sampling_params, "logprob_token_ids", None) is not None:
            raise ValueError("rollout.topk_log_probs requires the global top-k head, not custom logprob_token_ids")
        if sampling_params.temperature == 0:
            raise ValueError(
                "rollout.topk_log_probs requires stochastic sampling with temperature > 0; "
                "greedy processed log-probs are not the sampling distribution"
            )


def warn_topk_logprob_budget(sampling_params, warned_signatures: set[tuple]) -> None:
    """Warn once per effective request profile when the recorded head is below sampling top-k.

    This is a record-budget diagnostic, not evidence about support. Top-p/min-p may still
    shrink the actual support enough to fit, in which case provider coverage stays complete.
    """
    if (
        sampling_params.logprobs is None
        or sampling_params.logprobs <= 0
        or sampling_params.top_k <= sampling_params.logprobs
    ):
        return
    signature = (
        sampling_params.temperature,
        sampling_params.top_p,
        sampling_params.top_k,
        sampling_params.min_p,
        sampling_params.logprobs,
    )
    if signature not in warned_signatures:
        logger.warning(
            "The recorded head size %s is smaller than the effective sampling top_k=%s; "
            "increase rollout.topk_log_probs to record more sampler tokens. "
            "Actual support may still fit after top_p/min_p truncation. "
            "Effective request: temperature=%s, top_p=%s, min_p=%s.",
            sampling_params.logprobs,
            sampling_params.top_k,
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.min_p,
        )
        warned_signatures.add(signature)


def extract_response_topk_logprobs_with_coverage(
    logprobs, k: int, vocab_size: int | None = None
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    """Extract a global ranked top-k head and evidence that its support is complete.

    Call only for the provider's global top-k, never custom ``logprob_token_ids``: that path
    can use -inf for missing-record padding. Real vLLM FlatLogprobs offsets and ranks are
    checked, so a short record cannot be mistaken for zero-probability slots. Without this
    provider layout the evidence is unknown (False), including legacy records.

    A returned -inf in a global top-k proves that every omitted token also has probability
    zero. Recording the entire model vocabulary also proves completeness. An all-finite
    partial head remains unknown even if its probability mass rounds to one. Returns the
    usual sampled log-probs, ids and head log-probs, plus a bool array of shape (T,).
    """
    sampled, ids, head_log_probs = extract_response_topk_logprobs(logprobs, k)
    complete = np.zeros(len(sampled), dtype=np.bool_)
    sampled_ids = np.asarray(logprobs.token_ids[:: k + 1], dtype=np.int32)
    if vocab_size is not None:
        if vocab_size <= 0 or np.any(ids >= vocab_size) or np.any(sampled_ids >= vocab_size):
            raise ValueError("sampler top-k ids and sampled ids must lie within the model vocabulary")
    if not all(hasattr(logprobs, key) for key in ("start_indices", "end_indices", "ranks")):
        return sampled, ids, head_log_probs, complete
    ranks = np.asarray(logprobs.ranks).reshape(len(sampled), k + 1)
    if not np.all(ranks[:, 1:] == np.arange(1, k + 1)):
        raise ValueError("sampler top-k entries must have consecutive provider ranks 1..k")
    if k == vocab_size:
        complete[:] = True
    complete |= np.isneginf(head_log_probs).any(-1)
    sampled_in_head = ((ids == sampled_ids[:, None]) & np.isfinite(head_log_probs)).any(-1)
    if np.any(complete & ~sampled_in_head):
        raise ValueError("a complete sampler head must contain the sampled token with finite log-prob")
    if not np.allclose(np.exp(head_log_probs[complete]).sum(-1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("a complete sampler head must have unit probability mass")
    return sampled, ids, head_log_probs, complete


def _get_rollout_targets(config_file: str, server_addresses: list[str]) -> list[str]:
    """Merge new rollout server addresses into the existing Prometheus rollout targets."""
    try:
        with open(config_file) as f:
            existing_config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        existing_config = {}

    existing_targets: list[str] = []
    if isinstance(existing_config, dict):
        for scrape_config in existing_config.get("scrape_configs", []):
            if not isinstance(scrape_config, dict) or scrape_config.get("job_name") != "rollout":
                continue
            for static_config in scrape_config.get("static_configs", []):
                if isinstance(static_config, dict):
                    existing_targets.extend(static_config.get("targets", []))

    return list(dict.fromkeys([*existing_targets, *server_addresses]))


def update_prometheus_config(config: PrometheusConfig, server_addresses: list[str], rollout_name: str | None = None):
    """
    Update Prometheus configuration file with server addresses and reload on first node.
    Existing rollout targets in the configuration file are preserved.

    server_addresses: vllm or sglang server addresses

    rollout_name: name of the rollout backend (e.g., "vllm", "sglang")
    """

    if not server_addresses:
        logger.warning("No server addresses available to update Prometheus config")
        return

    try:
        # Get Prometheus config file path from environment or use default
        rollout_targets = _get_rollout_targets(config.file, server_addresses)
        prometheus_config_json = {
            "global": {"scrape_interval": "10s", "evaluation_interval": "10s"},
            "scrape_configs": [
                {
                    "job_name": "ray",
                    "file_sd_configs": [{"files": ["/tmp/ray/prom_metrics_service_discovery.json"]}],
                },
                {"job_name": "rollout", "static_configs": [{"targets": rollout_targets}]},
            ],
        }

        # Write configuration file to all nodes
        @ray.remote(num_cpus=0)
        def write_config_file(config_data, config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            return True

        # Reload Prometheus on all nodes. Only master node should succeed, skip errors on other nodes.
        @ray.remote(num_cpus=0)
        def reload_prometheus(port):
            import socket
            import subprocess

            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)

            reload_url = f"http://{ip_address}:{port}/-/reload"

            try:
                subprocess.run(["curl", "-X", "POST", reload_url], capture_output=True, text=True, timeout=10)
                print(f"Reloading Prometheus on node: {reload_url}")
            except Exception:
                # Skip errors on non-master nodes
                pass

        # Get all available nodes and schedule tasks on each node
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node["Alive"]]

        # Write config files on all nodes
        write_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = write_config_file.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(prometheus_config_json, config.file)
            write_tasks.append(task)

        ray.get(write_tasks)

        server_type = rollout_name.upper() if rollout_name else "rollout"
        print(f"Updated Prometheus configuration at {config.file} with {len(server_addresses)} {server_type} servers")

        # Reload Prometheus on all nodes
        reload_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = reload_prometheus.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(config.port)
            reload_tasks.append(task)

        ray.get(reload_tasks)

    except Exception as e:
        logger.error(f"Failed to update Prometheus configuration: {e}")
