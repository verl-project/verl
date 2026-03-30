"""HTTP client for the Atropos Trajectory API."""

import time

import requests


class AtroposClient:
    """HTTP client for the Atropos trajectory API with retry/backoff."""

    def __init__(self, api_url: str = "http://localhost:8000", max_retries: int = 3, retry_delay: float = 1.0):
        self.api_url = api_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _request(self, method: str, endpoint: str, **kwargs):
        """Make an HTTP request with exponential backoff on retryable errors."""
        url = f"{self.api_url}{endpoint}"
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                resp = getattr(requests, method)(url, **kwargs)
                resp.raise_for_status()
                return resp.json()
            except requests.ConnectionError:
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                if e.response is not None:
                    detail = f"{e.response.status_code} {e.response.text[:200]}"
                else:
                    detail = str(e)
                raise requests.HTTPError(f"Atropos API error on {method.upper()} {endpoint}: {detail}") from e

    def register_trainer(
        self,
        batch_size: int,
        max_token_len: int,
        num_steps: int,
        checkpoint_dir: str,
        save_checkpoint_interval: int = 1,
        starting_step: int = 0,
        wandb_group: str = "default",
        wandb_project: str = "verl-atropos",
    ) -> dict:
        """Register this trainer with the trajectory API."""
        payload = {
            "wandb_group": wandb_group,
            "wandb_project": wandb_project,
            "batch_size": batch_size,
            "max_token_len": max_token_len,
            "checkpoint_dir": checkpoint_dir,
            "save_checkpoint_interval": save_checkpoint_interval,
            "starting_step": starting_step,
            "num_steps": num_steps,
        }
        return self._request("post", "/register", json=payload)

    def get_batch(self) -> list[dict] | None:
        """Fetch a single batch. Returns list of ScoredData dicts, or None."""
        result = self._request("get", "/batch")
        batch = result.get("batch") if isinstance(result, dict) else result
        if batch is None:
            return None
        return batch

    def poll_batch(self, timeout: float = 300.0, poll_interval: float = 2.0) -> list[dict]:
        """Poll until a batch is available. Returns exactly one server-side batch."""
        start = time.time()

        while True:
            batch = self.get_batch()
            if batch is not None:
                return batch

            elapsed = time.time() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"No batch received from Atropos API at {self.api_url} "
                    f"within {timeout}s. Check that the environment server is "
                    f"running and pushing ScoredData to the trajectory API."
                )

            time.sleep(poll_interval)
