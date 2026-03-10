"""
verl <-> Atropos Reflex
Three functions that wire verl's skeleton to Atropos's skeleton.
"""
import logging
import time
import requests
import numpy as np
from typing import Optional, List

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from omegaconf import OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

try:
    from verl import DataProto
    HAS_VERL = True
except ImportError:
    HAS_VERL = False

logger = logging.getLogger(__name__)


def register_with_atropos(atropos_url: str, config, vllm_endpoints: List[str]) -> str:
    """Hook 1: fires after init_workers. POSTs Registration to Atropos."""
    if HAS_OMEGACONF:
        cfg = OmegaConf.to_container(config, resolve=True)
    else:
        cfg = config if isinstance(config, dict) else {}
    payload = {
        "wandb_group": cfg.get("trainer", {}).get("project_name", "verl_atropos"),
        "wandb_project": cfg.get("trainer", {}).get("project_name", "verl_atropos"),
        "batch_size": cfg.get("data", {}).get("train_batch_size", 32),
        "max_token_len": cfg.get("data", {}).get("max_prompt_length", 512) + cfg.get("data", {}).get("max_response_length", 1024),
        "checkpoint_dir": cfg.get("trainer", {}).get("default_hdfs_dir", "/tmp/verl_atropos"),
        "save_checkpoint_interval": cfg.get("trainer", {}).get("save_freq", 20),
        "starting_step": 0,
        "num_steps": cfg.get("trainer", {}).get("total_epochs", 10) * 1000,
        "inference_server_urls": vllm_endpoints,
    }
    resp = requests.post(f"{atropos_url}/register", json=payload, timeout=10).json()
    uuid = resp.get("uuid")
    logger.info(f"[Atropos] Registered — UUID: {uuid}")
    return uuid


def poll_batch(atropos_url: str) -> Optional[dict]:
    """Hook 2: fires when _get_gen_batch is called. GETs ScoredData from Atropos."""
    for attempt in range(30):
        try:
            resp = requests.get(f"{atropos_url}/batch", timeout=10).json()
            batch = resp.get("batch")
            if batch:
                return batch
            logger.debug(f"[Atropos] No batch ready, attempt {attempt + 1}/30")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[Atropos] Connection error attempt {attempt + 1}: {e}")
        except requests.exceptions.Timeout:
            logger.warning(f"[Atropos] Timeout attempt {attempt + 1}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[Atropos] Unrecoverable error: {e}")
            return None
        time.sleep(2)
    logger.error("[Atropos] Timed out after 30 attempts")
    return None


def scored_data_to_dataproto(batch: dict):
    """Hook 3: converts ScoredData fields to verl DataProto or numpy dict."""
    if HAS_TORCH:
        tokens = torch.tensor(batch["tokens"], dtype=torch.long)
        masks = torch.tensor(batch["masks"], dtype=torch.long)
        scores = torch.tensor(batch["scores"], dtype=torch.float)
        mean = scores.mean()
        std = scores.std() + 1e-8
        advantages = (scores - mean) / std
        if batch.get("advantages") is not None:
            advantages = torch.tensor(batch["advantages"], dtype=torch.float)
        if HAS_VERL:
            return DataProto.from_dict({
                "input_ids": tokens,
                "attention_mask": masks,
                "token_level_scores": scores,
                "token_level_advantages": advantages,
            })
        return {"input_ids": tokens, "attention_mask": masks, "token_level_scores": scores, "token_level_advantages": advantages}
    else:
        tokens = np.array(batch["tokens"], dtype=np.int64)
        masks = np.array(batch["masks"], dtype=np.int64)
        scores = np.array(batch["scores"], dtype=np.float32)
        mean = scores.mean()
        std = scores.std() + 1e-8
        advantages = (scores - mean) / std
        if batch.get("advantages") is not None:
            advantages = np.array(batch["advantages"], dtype=np.float32)
        return {"input_ids": tokens, "attention_mask": masks, "token_level_scores": scores, "token_level_advantages": advantages}
