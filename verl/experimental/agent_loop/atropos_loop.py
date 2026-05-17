import asyncio
import logging
import uuid
import numpy as np
import torch
from typing import Any, Optional, List, Dict
from pydantic import BaseModel

import httpx
from omegaconf import DictConfig, OmegaConf
from verl.protocol import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics

logger = logging.getLogger(__name__)

class AtroposClient:
    """
    Client for interacting with the Atropos environment microservice.
    """
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def register(self, registration_data: dict):
        """Register the trainer with the Atropos server."""
        resp = await self.client.post(f"{self.base_url}/register", json=registration_data)
        resp.raise_for_status()
        return resp.json()

    async def register_env(self, env_data: dict):
        """Register a specific environment."""
        resp = await self.client.post(f"{self.base_url}/register-env", json=env_data)
        resp.raise_for_status()
        return resp.json()

    async def get_batch(self):
        """Fetch the next batch of tasks from Atropos."""
        resp = await self.client.get(f"{self.base_url}/batch")
        resp.raise_for_status()
        return resp.json()

    async def submit_scored_data(self, scored_data: dict):
        """Submit scored rollout data to Atropos."""
        resp = await self.client.post(f"{self.base_url}/scored_data", json=scored_data)
        resp.raise_for_status()
        return resp.json()

    async def submit_scored_data_list(self, scored_data_list: List[dict]):
        """Submit a list of scored rollout data to Atropos."""
        resp = await self.client.post(f"{self.base_url}/scored_data_list", json=scored_data_list)
        resp.raise_for_status()
        return resp.json()

    async def get_status(self, env_id: int):
        """Get status for a specific environment."""
        resp = await self.client.get(f"{self.base_url}/status-env", params={"env_id": env_id})
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self.client.aclose()

class AtroposAgentLoopManager(AgentLoopBase):
    """
    Atropos-integrated Agent Loop Manager.
    
    Instead of using verl's internal dataset, this manager orchestrates rollouts
    by interacting with the Atropos microservice.
    """
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: Any,
        tokenizer: Any,
        processor: Any,
        dataset_cls: Any,
        data_config: DictConfigWrap,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, **kwargs)
        
        # Atropos configuration
        atropos_cfg = self.config.actor_rollout_ref.rollout.agent.get("atropos", {})
        self.atropos_url = atropos_cfg.get("url", "http://localhost:8000")
        self.client = AtroposClient(self.atropos_url)
        
        # Registration state
        self.registered_uuid = None
        self.env_id = None

    async def _ensure_registered(self):
        if self.registered_uuid is None:
            reg_data = {
                "wandb_group": self.config.trainer.project_name, # Simplification
                "wandb_project": self.config.trainer.experiment_name,
                "batch_size": self.config.actor_rollout_ref.rollout.gen_batch_size,
                "max_token_len": self.config.actor_rollout_ref.rollout.prompt_length + self.config.actor_rollout_ref.rollout.response_length,
                "checkpoint_dir": self.config.trainer.default_local_dir,
                "save_checkpoint_interval": self.config.trainer.save_freq,
                "starting_step": 0, # Will be updated by trainer
                "num_steps": self.config.trainer.total_training_steps,
            }
            res = await self.client.register(reg_data)
            self.registered_uuid = res.get("uuid")
            
            # Register a default environment if not specified
            env_data = {
                "max_token_length": self.config.actor_rollout_ref.rollout.prompt_length + self.config.actor_rollout_ref.rollout.response_length,
                "desired_name": "verl_default_env",
                "weight": 1.0,
                "group_size": self.config.actor_rollout_ref.rollout.n,
            }
            env_res = await self.client.register_env(env_data)
            self.env_id = env_res.get("env_id")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Overridden run method.
        
        In a standard verl loop, this is called per-sample.
        With Atropos, we might want to handle batches.
        For compatibility with the existing AgentLoopWorker, we implement the per-sample logic,
        but the Atropos integration primarily affects how data is fetched (RayPPOTrainer).
        """
        # Current implementation of AgentLoopBase expects to be called per sample.
        # If we are using Atropos, the RayPPOTrainer will be fetching batches from Atropos
        # and passing them here.
        
        # For now, we implement a basic single-turn response since the primary 
        # Atropos integration happens at the Trainer level for batch orchestration.
        
        # Construct the prompt from kwargs (which contain dataset fields)
        messages = [{"role": "user", "content": kwargs.get("raw_prompt", "")}]
        prompt_ids = await self.apply_chat_template(messages)
        
        # Generate response using the LLM server manager
        # Note: The server_manager is an LLMServerClient
        response = await self.server_manager.generate(
            uuid.uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params
        )
        
        # response is typically a dict with "tokens" and "logprobs"
        response_ids = response.get("tokens", [])
        response_logprobs = response.get("logprobs", [])
        
        # We need to compute the score (reward)
        # This would normally be done by the reward model.
        # In Atropos, the environment provides the reward.
        reward_score = kwargs.get("reward_score", 0.0) # Default or from data
        
        # Construct the AgentLoopOutput
        metrics = AgentLoopMetrics()
        
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            reward_score=reward_score,
            num_turns=1,
            metrics=metrics
        )

    async def __aenter__(self):
        await self._ensure_registered()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()
