"""
Atropos Async Rollout Manager for verl.

This module provides an async rollout manager that integrates with verl's Ray trainer.
It replaces the internal vLLM-based rollout with Atropos API-based rollouts.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import ray
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.torch_functional import masked_mean

if TYPE_CHECKING:
    from transformers import AutoTokenizer
    from verl.workers.rollout.vllm_rollout import VLLMRollout

logger = logging.getLogger(__name__)


@ray.remote
class AtroposRolloutActor:
    """
    Ray actor that handles Atropos rollouts on a single GPU.

    Responsible for:
    1. Communicating with Atropos API
    2. Generating responses via vLLM
    3. Submitting scored data back to Atropos
    """

    def __init__(
        self,
        atropos_url: str,
        vllm_model_path: str,
        vllm_port: int,
        gpu_id: int,
        max_token_len: int = 1024,
        batch_size: int = 16,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        tokenizer_path: Optional[str] = None,
        trust_remote_code: bool = True,
        config: Optional[Dict] = None,
    ):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.atropos_url = atropos_url
        self.vllm_model_path = vllm_model_path
        self.vllm_port = vllm_port
        self.gpu_id = gpu_id
        self.max_token_len = max_token_len
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tokenizer_path = tokenizer_path or vllm_model_path
        self.trust_remote_code = trust_remote_code
        self.config = config or {}

        self._tokenizer = None
        self._vllm_process = None
        self._atropos_uuid = None

    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=self.trust_remote_code,
            )
        return self._tokenizer

    def initialize(self):
        """Initialize vLLM server and Atropos connection."""
        from vllm.entrypoints.openai.api_server import VLLM_API_SERVER
        import subprocess
        import requests

        # Start vLLM server
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.vllm_model_path,
            "--port", str(self.vllm_port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--trust-remote-code",
        ]

        logger.info(f"Starting vLLM on GPU {self.gpu_id}: {' '.join(cmd)}")
        self._vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for vLLM to be ready
        vllm_ready = False
        for _ in range(60):
            try:
                resp = requests.get(f"http://localhost:{self.vllm_port}/health", timeout=2)
                if resp.status_code == 200:
                    vllm_ready = True
                    break
            except (requests.RequestException, OSError):
                pass
            time.sleep(2)

        if not vllm_ready:
            raise RuntimeError(f"vLLM server failed to start on port {self.vllm_port}")

        logger.info(f"vLLM ready on GPU {self.gpu_id}")

        # Register with Atropos
        self._register_with_atropos()

    def _register_with_atropos(self):
        """Register with Atropos API."""
        import requests

        response = requests.post(
            f"{self.atropos_url}/register",
            json={
                "wandb_group": "",
                "wandb_project": "verl-atropos",
                "batch_size": self.batch_size,
                "max_token_len": self.max_token_len,
                "checkpoint_dir": "./checkpoints",
                "save_checkpoint_interval": 100,
                "starting_step": 0,
                "num_steps": 10000,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        self._atropos_uuid = data.get("uuid")
        logger.info(f"Registered with Atropos (uuid: {self._atropos_uuid})")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate_sequences(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        max_new_tokens: int = 256,
        do_sample: bool = True,
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Generate sequences using vLLM.

        Returns:
            Tuple of (generated_texts, token_logprobs)
        """
        import requests

        endpoint = f"http://localhost:{self.vllm_port}/v1/chat/completions"
        messages = [{"role": "user", "content": p} for p in prompts]

        payload = {
            "model": self.vllm_model_path,
            "messages": messages,
            "temperature": temperature if do_sample else 0.0,
            "max_tokens": max_new_tokens,
            "logprobs": True,
            "top_logprobs": 5,
        }

        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        outputs = data["choices"]
        generated_texts = [o["message"]["content"] for o in outputs]

        logprob_sequences = []
        for o in outputs:
            if "logprobs" in o and o["logprobs"] is not None:
                token_logprobs = [
                    tok.get("logprob", 0.0)
                    for tok in o["logprobs"]["content"]
                ]
                logprob_sequences.append(token_logprobs)
            else:
                logprob_sequences.append([])

        return generated_texts, logprob_sequences

    def get_batch_from_atropos(self, timeout: float = 60.0) -> Optional[Dict[str, Any]]:
        """Get a batch of prompts from Atropos."""
        import requests

        try:
            response = requests.get(
                f"{self.atropos_url}/batch",
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "error":
                return None
            return data
        except Exception as e:
            logger.warning(f"Failed to get batch from Atropos: {e}")
            return None

    def submit_to_atropos(
        self,
        tokens: List[List[int]],
        masks: List[List[int]],
        scores: List[float],
        logprobs: Optional[List[List[float]]] = None,
        advantages: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """Submit scored data to Atropos."""
        import requests

        payload = {
            "tokens": tokens,
            "masks": masks,
            "scores": scores,
        }
        if logprobs is not None:
            payload["inference_logprobs"] = logprobs
        if advantages is not None:
            payload["advantages"] = advantages

        response = requests.post(
            f"{self.atropos_url}/scored_data",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def update_weights(self, state_dict: Dict[str, Any]):
        """
        Update model weights from trainer.

        For the Atropos integration, weights are managed by the trainer
        and the vLLM server needs to be restarted to pick up new weights.
        This is a placeholder for future weight-bridge integration.
        """
        pass

    def shutdown(self):
        """Clean up resources."""
        if self._vllm_process:
            self._vllm_process.terminate()
            self._vllm_process.wait(timeout=10)


class AtroposRolloutManager:
    """
    Manages Atropos-based rollouts for the verl trainer.

    This manager:
    1. Creates and manages AtroposRolloutActor workers on Ray
    2. Coordinates batch distribution and collection
    3. Handles async rollout operations
    """

    def __init__(
        self,
        config: Dict[str, Any],
        worker_group: Optional[RayWorkerGroup] = None,
    ):
        """
        Initialize the Atropos rollout manager.

        Args:
            config: Configuration dict with keys:
                - atropos_url: Atropos API server URL
                - model_path: Model path for vLLM
                - vllm_port: Starting port for vLLM servers
                - batch_size: Training batch size
                - max_token_len: Maximum token length
                - gpu_memory_utilization: GPU memory fraction for vLLM
                - n: Number of rollouts per sample
            worker_group: Optional Ray worker group (for actor placement)
        """
        self.config = config
        self.worker_group = worker_group
        self.actors = []

        self.atropos_url = config.get("atropos_url", "http://localhost:8000")
        self.model_path = config.get("model_path", "Qwen/Qwen2.5-3B-Instruct")
        self.vllm_port_start = config.get("vllm_port", 8100)
        self.batch_size = config.get("batch_size", 16)
        self.max_token_len = config.get("max_token_len", 1024)
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.8)
        self.n_rollouts = config.get("n", 1)

    def create_actors(self, num_gpus: int):
        """Create AtroposRolloutActor workers on GPUs."""
        from verl.workers.rollout.vllm_rollout import VLLMRollout

        for gpu_id in range(num_gpus):
            actor = AtroposRolloutActor.options(
                num_cpus=4,
                num_gpus=1,
                resources={f"GPU": 1} if False else None,
            ).remote(
                atropos_url=self.atropos_url,
                vllm_model_path=self.model_path,
                vllm_port=self.vllm_port_start + gpu_id,
                gpu_id=gpu_id,
                max_token_len=self.max_token_len,
                batch_size=self.batch_size // num_gpus,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.gpu_memory_utilization,
                config=self.config,
            )
            self.actors.append(actor)

        # Initialize actors
        ray.get([actor.initialize.remote() for actor in self.actors])
        logger.info(f"Created {len(self.actors)} Atropos rollout actors")

    def generate_sequences(self, batch: DataProto) -> DataProto:
        """
        Generate sequences for the given batch using Atropos.

        Args:
            batch: DataProto containing prompts

        Returns:
            DataProto with generated sequences and logprobs
        """
        prompts = batch.non_tensor_batch.get("prompts", [])

        if not prompts:
            return batch

        # Distribute prompts across actors
        prompts_per_actor = len(prompts) // len(self.actors) + 1
        futures = []

        for i, actor in enumerate(self.actors):
            start_idx = i * prompts_per_actor
            end_idx = min(start_idx + prompts_per_actor, len(prompts))
            actor_prompts = prompts[start_idx:end_idx]

            if actor_prompts:
                futures.append(
                    actor.generate_sequences.remote(
                        prompts=actor_prompts,
                        temperature=batch.meta_info.get("temperature", 1.0),
                        max_new_tokens=self.max_token_len,
                        do_sample=True,
                    )
                )

        # Collect results
        results = ray.get(futures)

        all_responses = []
        all_logprobs = []

        for responses, logprobs in results:
            all_responses.extend(responses)
            all_logprobs.extend(logprobs)

        # Update batch with results
        batch.non_tensor_batch["responses"] = all_responses
        batch.meta_info["logprobs"] = all_logprobs

        return batch

    def update_weights(self, state_dict: Dict[str, Any]):
        """Update weights across all actors."""
        futures = [actor.update_weights.remote(state_dict) for actor in self.actors]
        ray.get(futures)

    def shutdown(self):
        """Shutdown all actors."""
        if self.actors:
            ray.get([actor.shutdown.remote() for actor in self.actors])
            self.actors = []
