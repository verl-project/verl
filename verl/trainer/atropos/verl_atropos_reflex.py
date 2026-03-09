"""
verl <-> Atropos Reflex Layer
Connects verl's RayPPOTrainer (GRPO mode) to Atropos RL environment API.
Subclasses RayPPOTrainer — drop-in replacement with Atropos as data source.
"""
import requests
import time
import torch
from typing import Optional
from omegaconf import DictConfig, open_dict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


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
        rollout_cfg = cfg.get("actor_rollout_ref", {}).get("rollout", {})

        # Collect vLLM inference server endpoints spun up by verl
        endpoints = self._get_vllm_endpoints()

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
            print(f"[Atropos] Registered — UUID: {self._atropos_uuid}")
        except requests.exceptions.RequestException as e:
            print(f"[Atropos] Registration failed: {e}")

    def _get_vllm_endpoints(self):
        """Get vLLM inference server URLs from Ray worker group."""
        try:
            endpoints = []
            for worker in self.actor_rollout_wg.get_workers():
                import ray
                host = ray.get(worker.get_host.remote())
                port = ray.get(worker.get_vllm_port.remote())
                endpoints.append(f"http://{host}:{port}")
            return endpoints
        except Exception as e:
            print(f"[Atropos] Error getting vLLM endpoints: {e}")
            return []

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
            except requests.exceptions.RequestException as e:
                print(f"[Atropos] Failed to get batch on attempt {attempt + 1}: {e}")
            time.sleep(2)
        print("[Atropos] Timed out waiting for batch")
        return None

    def _atropos_batch_to_dataproto(self, batch: dict) -> DataProto:
        """Convert Atropos batch format to verl DataProto."""
        import torch
        tokens = torch.tensor(batch.get("tokens", []), dtype=torch.long)
        masks = torch.tensor(batch.get("masks", []), dtype=torch.long)
        scores = torch.tensor(batch.get("scores", []), dtype=torch.float)

        data = DataProto.from_dict({
            "input_ids": tokens,
            "attention_mask": masks,
            "token_level_scores": scores,
        })

        # Token-level advantage override if provided
        if "advantages" in batch:
            advantages = torch.tensor(batch["advantages"], dtype=torch.float)
            data.batch["token_level_advantages"] = advantages

        return data

    def fit(self):
        """
        Training loop — pulls batches from Atropos instead of parquet files.
        Uses GRPO advantage estimation unless Atropos provides token-level advantages.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        logger = Tracking(
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
                break

            # 2. Compute advantages (GRPO) unless Atropos provided token-level
            if "token_level_advantages" not in batch.batch:
                batch = self._compute_grpo_advantages(batch)

            # 3. Train
            metrics = self._update_actor(batch)

            # 4. Push updated weights to vLLM inference servers
            self._sync_weights_to_vllm()

            # 5. Log
            logger.log(metrics, step=step)

            if step % self.config.trainer.get("save_freq", 20) == 0:
                self._save_checkpoint(step)

    def _compute_grpo_advantages(self, batch: DataProto) -> DataProto:
        """Compute GRPO advantages from token-level scores."""
        scores = batch.batch["token_level_scores"]
        # GRPO: normalize scores within group
        mean = scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True) + 1e-8
        batch.batch["token_level_advantages"] = (scores - mean) / std
        return batch

    def _sync_weights_to_vllm(self):
        """Push updated policy weights to vLLM inference servers."""
        try:
            self.actor_rollout_wg.sync_model_weights()
        except Exception as e:
            print(f"[Atropos] Weight sync failed: {e}")
