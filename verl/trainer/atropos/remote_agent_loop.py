"""
RemoteAgentLoop — verl integration for external distributed agents.

Satisfies #5737: enables RL training of agents running in separate
Ray/Kubernetes clusters without modifying the agent code.

Connects external agents to verl via the Atropos trajectory API:
- register_with_atropos() — registers verl endpoints with Atropos
- poll_batch() — pulls scored trajectories from external agents
- scored_data_to_dataproto() — converts to verl training format

Unlike ToolAgentLoop and ReactAgentLoop, RemoteAgentLoop:
- Runs agents in isolated external environments (Docker, K8s, Ray)
- Captures token-level data via HTTP proxy recording
- Requires zero modifications to the external agent
- Works with any OpenAI-compatible agent framework
"""
import logging
import time
import requests
from typing import Optional, Any

logger = logging.getLogger(__name__)


class RemoteAgentLoop:
    """
    AgentLoop that pulls pre-scored rollouts from an external agent
    running via Atropos, rather than generating rollouts internally.

    Usage:
        loop = RemoteAgentLoop(atropos_url="http://atropos-server:8000")
        loop.register(config, vllm_endpoints)
        batch = loop.get_batch()
        dataproto = loop.to_dataproto(batch)
    """

    def __init__(
        self,
        atropos_url: str = "http://localhost:8000",
        poll_timeout: float = 300.0,
        poll_interval: float = 2.0,
        max_retries: int = 3,
    ):
        self.atropos_url = atropos_url.rstrip("/")
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.uuid: Optional[str] = None
        self._session = requests.Session()

    def register(self, config: Any, vllm_endpoints: list[str]) -> str:
        """
        Register verl with Atropos after workers initialize.
        Maps to Hook 1: register_with_atropos()
        """
        from .verl_atropos_reflex import register_with_atropos
        self.uuid = register_with_atropos(self.atropos_url, config, vllm_endpoints)
        return self.uuid

    def get_batch(self) -> Optional[dict]:
        """
        Poll Atropos for a scored trajectory batch from the external agent.
        Maps to Hook 2: poll_batch()

        Returns None if no batch available within poll_timeout.
        """
        deadline = time.time() + self.poll_timeout
        attempt = 0

        while time.time() < deadline:
            try:
                resp = self._session.get(
                    f"{self.atropos_url}/batch",
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                batch = data.get("batch")
                if batch:
                    logger.info(f"[RemoteAgentLoop] Got batch with {len(batch.get('scores', []))} trajectories")
                    return batch
                logger.debug(f"[RemoteAgentLoop] No batch ready, waiting {self.poll_interval}s...")
            except requests.exceptions.ConnectionError as e:
                attempt += 1
                if attempt >= self.max_retries:
                    logger.error(f"[RemoteAgentLoop] Connection failed after {self.max_retries} retries: {e}")
                    return None
                logger.warning(f"[RemoteAgentLoop] Connection error (retry {attempt}/{self.max_retries}): {e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"[RemoteAgentLoop] Unrecoverable error: {e}")
                return None

            time.sleep(self.poll_interval)

        logger.error(f"[RemoteAgentLoop] Timed out after {self.poll_timeout}s")
        return None

    def to_dataproto(self, batch: dict):
        """
        Convert Atropos scored batch to verl DataProto for GRPO training.
        Maps to Hook 3: scored_data_to_dataproto()
        """
        from .verl_atropos_reflex import scored_data_to_dataproto
        return scored_data_to_dataproto(batch)

    def run_cycle(self, config: Any, vllm_endpoints: list[str]):
        """
        Full training cycle: register → poll → convert.
        Returns DataProto ready for GRPO advantage computation.
        """
        if self.uuid is None:
            self.register(config, vllm_endpoints)

        batch = self.get_batch()
        if batch is None:
            return None

        return self.to_dataproto(batch)

    @classmethod
    def from_config(cls, config: Any) -> "RemoteAgentLoop":
        """
        Construct from verl config object.

        Expected config keys (under actor_rollout_ref.rollout.remote_agent):
            atropos_url: str
            poll_timeout: float
            poll_interval: float
            max_retries: int
        """
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.to_container(config, resolve=True)
        except Exception:
            cfg = config if isinstance(config, dict) else {}

        remote_cfg = cfg.get("actor_rollout_ref", {}).get("rollout", {}).get("remote_agent", {})

        return cls(
            atropos_url=remote_cfg.get("atropos_url", "http://localhost:8000"),
            poll_timeout=remote_cfg.get("poll_timeout", 300.0),
            poll_interval=remote_cfg.get("poll_interval", 2.0),
            max_retries=remote_cfg.get("max_retries", 3),
        )
