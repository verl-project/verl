"""
verl <-> Atropos Reflex Layer
Connects verl's RayPPOTrainer (GRPO mode) to Atropos RL environment API.
Subclasses RayPPOTrainer — drop-in replacement with Atropos as data source.
"""
import logging
import requests
import numpy as np
import time
import torch
from typing import Optional, List
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.tracking import Tracking

logger = logging.getLogger(__name__)


class AtroposVerlTrainer(RayPPOTrainer):
    """
    RayPPOTrainer subclass that sources rollout data from Atropos
    instead of a static dataset. Uses GRPO advantage estimation.
    Atropos manages environments; verl manages inference + training.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        atropos_cfg = config.get("atropos", {})
        host = atropos_cfg.get("host", "localhost")
        port = atropos_cfg.get("port", 8000)
        if not host.startswith("http"):
            host = f"http://{host}"
        self.atropos_url = f"{host}:{port}"
        self._atropos_uuid = None

    def init_workers(self):
        """Initialize Ray workers then register with Atropos."""
        super().init_workers()
        self._register_with_atropos()

    def _register_with_atropos(self):
        """POST /register — tell Atropos our batch size and vLLM endpoints."""
        cfg = self.config
        trainer_cfg = cfg.get("trainer", {})

        endpoints = self._get_vllm_endpoints()
        if not endpoints:
            logger.warning("[Atropos] No vLLM endpoints found — registering without inference servers")

        register_data = {
            "wandb_group": trainer_cfg.get("project_name", "verl_atropos"),
            "wandb_project": trainer_cfg.get("project_name", "verl_atropos"),
            "batch_size": cfg.get("data", {}).get("train_batch_size", 32),
            "max_token_len": cfg.get("data", {}).get("max_prompt_length", 512)
                           + cfg.get("data", {}).get("max_response_length", 1024),
            "checkpoint_dir": trainer_cfg.get("default_hdfs_dir", "/tmp/verl_atropos"),
            "save_checkpoint_interval": trainer_cfg.get("save_freq", 20),
            "starting_step": 0,
            "num_steps": trainer_cfg.get("total_epochs", 10) * 1000,
            "inference_server_urls": endpoints,
        }

        try:
            resp = requests.post(
                f"{self.atropos_url}/register",
                json=register_data,
                timeout=10,
            ).json()
            self._atropos_uuid = resp.get("uuid")
            logger.info(f"[Atropos] Registered — UUID: {self._atropos_uuid}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[Atropos] Registration failed: {e}")
            raise

    def _get_vllm_endpoints(self) -> List[str]:
        """Get vLLM inference server URLs from Ray worker group."""
        endpoints = []
        try:
            import ray
            for worker in self.actor_rollout_wg.get_workers():
                host = ray.get(worker.get_host.remote())
                port = ray.get(worker.get_vllm_port.remote())
                endpoints.append(f"http://{host}:{port}")
        except AttributeError as e:
            logger.warning(f"[Atropos] Worker group not ready for endpoint discovery: {e}")
        except Exception as e:
            logger.error(f"[Atropos] Unexpected error getting vLLM endpoints: {e}")
            raise
        return endpoints

    def _get_atropos_batch(self) -> Optional[DataProto]:
        """GET /batch — poll Atropos for scored trajectory data."""
        for attempt in range(30):
            try:
                resp = requests.get(
                    f"{self.atropos_url}/batch",
                    timeout=10,
                ).json()
                batch = resp.get("batch")
                if batch:
                    return self._atropos_batch_to_dataproto(batch)
                logger.debug(f"[Atropos] No batch ready on attempt {attempt + 1}, retrying in 2s")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[Atropos] Connection error on attempt {attempt + 1}: {e}")
            except requests.exceptions.Timeout:
                logger.warning(f"[Atropos] Timeout on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.error(f"[Atropos] Unrecoverable request error: {e}")
                return None
            time.sleep(2)
        logger.error("[Atropos] Timed out waiting for batch after 30 attempts (60s)")
        return None

    def _atropos_batch_to_dataproto(self, batch: dict) -> DataProto:
        """Convert Atropos batch format to verl DataProto."""
        tokens = torch.tensor(batch.get("tokens", []), dtype=torch.long)
        masks = torch.tensor(batch.get("masks", []), dtype=torch.long)
        scores = torch.tensor(batch.get("scores", []), dtype=torch.float)

        data = DataProto.from_dict({
            "input_ids": tokens,
            "attention_mask": masks,
            "token_level_scores": scores,
        })

        if "advantages" in batch:
            advantages = torch.tensor(batch["advantages"], dtype=torch.float)
            data.batch["token_level_advantages"] = advantages

        return data

    def fit(self):
        """
        Training loop — pulls batches from Atropos instead of parquet files.
        Uses GRPO advantage estimation unless Atropos provides token-level advantages.
        """
        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        logger.info("Starting Atropos-verl GRPO training loop")

        for step in range(self.total_training_steps):
            # 1. Get batch from Atropos
            batch = self._get_atropos_batch()
            if batch is None:
                logger.error(f"[Atropos] No batch at step {step}, stopping training")
                break

            # 2. Compute advantages (GRPO) unless Atropos provided token-level
            if "token_level_advantages" not in batch.batch:
                batch = self._compute_grpo_advantages(batch)

            # 3. Train
            metrics = self._update_actor(batch)

            # 4. Push updated weights to vLLM inference servers
            self._sync_weights_to_vllm()

            # 5. Log
            tracker.log(metrics, step=step)

            if step % self.config.trainer.get("save_freq", 20) == 0:
                self._save_checkpoint(step)

    def _compute_grpo_advantages(self, batch: DataProto) -> DataProto:
        """Compute GRPO advantages from token-level scores.
        Uses numpy for ARM-native computation when torch unavailable.
        """
        scores = batch.batch["token_level_scores"]
        try:
            mean = scores.mean(dim=-1, keepdim=True)
            std = scores.std(dim=-1, keepdim=True) + 1e-8
            batch.batch["token_level_advantages"] = (scores - mean) / std
        except Exception:
            s = np.array(scores) if not hasattr(scores, 'numpy') else scores.numpy()
            mean = s.mean()
            std = s.std() + 1e-8
            batch.batch["token_level_advantages"] = (s - mean) / std
        return batch

    def _sync_weights_to_vllm(self):
        """Push updated policy weights to vLLM inference servers."""
        try:
            self.actor_rollout_wg.sync_model_weights()
        except Exception as e:
            logger.warning(f"[Atropos] Weight sync failed: {e}")
