"""
Atropos API client for verl integration.

Handles communication with the Atropos API server:
- Trainer registration
- Batch retrieval (prompts)
- Scored data submission (tokens, scores, advantages)
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class AtroposAPIClient:
    """Client for communicating with Atropos API server."""

    def __init__(
        self,
        url: str = "http://localhost:8000",
        timeout: float = 30.0,
        trainer_uuid: Optional[str] = None,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.trainer_uuid = trainer_uuid
        self._vllm_port = None
        self._session = requests.Session()

    def check_server(self) -> bool:
        """Check if Atropos API server is reachable."""
        try:
            response = self._session.get(f"{self.url}/info", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_server(self, timeout: float = 60.0) -> bool:
        """Wait for Atropos API server to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            if self.check_server():
                logger.info(f"Atropos API server is ready at {self.url}")
                return True
            logger.debug(f"Waiting for Atropos API at {self.url}...")
            time.sleep(2)
        return False

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    def register(
        self,
        wandb_group: str = "",
        wandb_project: str = "",
        batch_size: int = 16,
        max_token_len: int = 1024,
        checkpoint_dir: str = "./checkpoints",
        save_checkpoint_interval: int = 100,
        starting_step: int = 0,
        num_steps: int = 1000,
    ) -> str:
        """
        Register this trainer with Atropos API.

        Returns the trainer UUID assigned by Atropos.
        """
        response = self._session.post(
            f"{self.url}/register",
            json={
                "wandb_group": wandb_group,
                "wandb_project": wandb_project,
                "batch_size": batch_size,
                "max_token_len": max_token_len,
                "checkpoint_dir": checkpoint_dir,
                "save_checkpoint_interval": save_checkpoint_interval,
                "starting_step": starting_step,
                "num_steps": num_steps,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if "uuid" not in data:
            raise RuntimeError(f"Registration failed: {data}")
        self.trainer_uuid = data["uuid"]
        logger.info(f"Registered with Atropos (uuid: {self.trainer_uuid})")
        return self.trainer_uuid

    def get_batch(self) -> Dict[str, Any]:
        """
        Get a batch of training prompts from Atropos.

        Returns a dict containing:
        - batch: List of prompts/data items
        - current_step: Current training step
        """
        response = self._session.get(f"{self.url}/batch", timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "error":
            raise RuntimeError(f"Atropos API error: {data.get('message', 'Unknown')}")
        return data

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def submit_scored_data(
        self,
        tokens: List[List[int]],
        masks: List[List[int]],
        scores: List[float],
        advantages: Optional[List[List[float]]] = None,
        ref_logprobs: Optional[List[List[float]]] = None,
        inference_logprobs: Optional[List[List[float]]] = None,
        overrides: Optional[List[dict]] = None,
        group_overrides: Optional[dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Submit scored rollout data to Atropos.

        Args:
            tokens: List of token sequences [batch_size, seq_len]
            masks: List of attention masks [batch_size, seq_len]
            scores: List of per-instance scores [batch_size]
            advantages: Optional token-level advantages [batch_size, seq_len]
            ref_logprobs: Optional reference log probabilities
            inference_logprobs: Optional inference log probabilities
            overrides: Optional per-instance overrides
            group_overrides: Optional group-level overrides

        Returns:
            Acknowledgment from Atropos API
        """
        payload = {
            "tokens": tokens,
            "masks": masks,
            "scores": scores,
        }
        if advantages is not None:
            payload["advantages"] = advantages
        if ref_logprobs is not None:
            payload["ref_logprobs"] = ref_logprobs
        if inference_logprobs is not None:
            payload["inference_logprobs"] = inference_logprobs
        if overrides is not None:
            payload["overrides"] = overrides
        if group_overrides is not None:
            payload["group_overrides"] = group_overrides

        payload.update(kwargs)

        response = self._session.post(
            f"{self.url}/scored_data",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_vllm_endpoint(self) -> str:
        """Get the vLLM server URL that Atropos should use for inference."""
        return f"http://localhost:{self._vllm_port}"

    def set_vllm_port(self, port: int):
        """Set the vLLM port for reference."""
        self._vllm_port = port

    def close(self):
        """Close the session."""
        self._session.close()
