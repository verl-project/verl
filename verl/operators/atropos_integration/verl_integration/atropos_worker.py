"""
Atropos Worker - Ray worker for Atropos-based rollouts.

This module provides a Ray-based worker that can be used with verl's
single-controller architecture.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch

from verl import DataProto
from verl.single_controller.ray.base import RayWorker

logger = logging.getLogger(__name__)


@ray.remote
class AtroposWorker(RayWorker):
    """
    Ray worker for Atropos rollouts.

    This worker handles:
    1. Communication with Atropos API
    2. Tokenization and de-tokenization
    3. vLLM server management for inference
    4. Data submission to Atropos
    """

    def __init__(
        self,
        atropos_url: str = "http://localhost:8000",
        vllm_model_path: str = "Qwen/Qwen2.5-3B-Instruct",
        vllm_port: int = 8100,
        max_token_len: int = 1024,
        batch_size: int = 16,
        gpu_id: int = 0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        tokenizer_path: Optional[str] = None,
        trust_remote_code: bool = True,
        config: Optional[Dict] = None,
    ):
        """
        Initialize Atropos worker.

        Args:
            atropos_url: URL of Atropos API server
            vllm_model_path: Model path for vLLM
            vllm_port: Port for vLLM server
            max_token_len: Maximum sequence length
            batch_size: Training batch size
            gpu_id: GPU ID to use
            tensor_parallel_size: Tensor parallelism size
            gpu_memory_utilization: GPU memory fraction for vLLM
            tokenizer_path: Path to tokenizer
            trust_remote_code: Trust remote code in tokenization
            config: Additional configuration
        """
        super().__init__()

        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.atropos_url = atropos_url
        self.vllm_model_path = vllm_model_path
        self.vllm_port = vllm_port
        self.max_token_len = max_token_len
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tokenizer_path = tokenizer_path or vllm_model_path
        self.trust_remote_code = trust_remote_code
        self.config = config or {}

        self._tokenizer = None
        self._vllm_process = None
        self._atropos_uuid = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of tokenizer and vLLM."""
        if self._initialized:
            return

        from transformers import AutoTokenizer
        import subprocess
        import requests

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=self.trust_remote_code,
        )

        # Start vLLM server
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.vllm_model_path,
            "--port", str(self.vllm_port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--trust-remote-code",
        ]

        logger.info(f"[Worker-{self.gpu_id}] Starting vLLM: {' '.join(cmd)}")
        self._vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for vLLM ready
        import time
        for _ in range(60):
            try:
                resp = requests.get(f"http://localhost:{self.vllm_port}/health", timeout=2)
                if resp.status_code == 200:
                    break
            except:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(f"vLLM failed to start on port {self.vllm_port}")

        # Register with Atropos
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
                "num_steps": 100000,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        self._atropos_uuid = data.get("uuid")

        self._initialized = True
        logger.info(f"[Worker-{self.gpu_id}] Initialized (Atropos uuid: {self._atropos_uuid})")

    def _generate_with_vllm(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        max_new_tokens: int = 256,
    ) -> Tuple[List[str], List[List[float]]]:
        """Generate responses using vLLM."""
        import requests

        endpoint = f"http://localhost:{self.vllm_port}/v1/chat/completions"
        messages = [{"role": "user", "content": p} for p in prompts]

        payload = {
            "model": self.vllm_model_path,
            "messages": messages,
            "temperature": temperature,
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
                token_logprobs = [tok.get("logprob", 0.0) for tok in o["logprobs"]["content"]]
                logprob_sequences.append(token_logprobs)
            else:
                logprob_sequences.append([])

        return generated_texts, logprob_sequences

    def generate_sequences(self, batch: DataProto) -> DataProto:
        """
        Generate sequences for prompts in the batch.

        Args:
            batch: DataProto with prompts in non_tensor_batch

        Returns:
            DataProto with generated responses and logprobs
        """
        self._lazy_init()

        prompts = batch.non_tensor_batch.get("prompts", [])
        temperature = batch.meta_info.get("temperature", 1.0)
        max_new_tokens = self.max_token_len

        if not prompts:
            return batch

        responses, logprobs = self._generate_with_vllm(
            prompts=prompts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Tokenize responses
        response_ids = []
        response_masks = []
        for resp in responses:
            ids = self._tokenizer.encode(resp, add_special_tokens=False)
            response_ids.append(ids)
            response_masks.append([1] * len(ids))

        # Pad sequences
        max_len = max(len(ids) for ids in response_ids)
        pad_token_id = self._tokenizer.pad_token_id or 0

        padded_ids = []
        padded_masks = []
        for ids, mask in zip(response_ids, response_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_token_id] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

        # Update batch
        batch.batch["responses"] = torch.tensor(padded_ids)
        batch.batch["response_mask"] = torch.tensor(padded_masks)
        batch.non_tensor_batch["response_texts"] = responses
        batch.meta_info["token_logprobs"] = logprobs

        return batch

    def get_batch_from_atropos(self, timeout: float = 60.0) -> Optional[Dict[str, Any]]:
        """Get a batch of training prompts from Atropos."""
        import requests

        try:
            response = requests.get(f"{self.atropos_url}/batch", timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "error":
                return None
            return data
        except Exception as e:
            logger.warning(f"[Worker-{self.gpu_id}] Failed to get Atropos batch: {e}")
            return None

    def submit_scored_data(
        self,
        tokens: List[List[int]],
        masks: List[List[int]],
        scores: List[float],
        logprobs: Optional[List[List[float]]] = None,
        advantages: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """Submit scored rollout data to Atropos."""
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
        Update model weights.

        For the Atropos integration, weights are managed by the
        main trainer and this is a no-op placeholder.
        """
        pass

    def shutdown(self):
        """Clean up resources."""
        if self._vllm_process:
            self._vllm_process.terminate()
            self._vllm_process.wait(timeout=10)
