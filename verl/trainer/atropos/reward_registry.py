"""
verl Atropos Reward Registry
Pluggable reward bridge — when verl cannot find a reward function for a
data source, registered handlers are tried in order. Falls back to
Atropos scoring if all else fails.

Usage:
    from verl.trainer.atropos.reward_registry import RewardRegistry
    registry = RewardRegistry(atropos_url="http://localhost:8000")
    score = registry.compute_score(data_source, solution_str, ground_truth)
"""

import logging
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Built-in handlers for common data sources that may be missing


def _gsm8k_score(solution_str: str, ground_truth: str) -> float:
    """GSM8K reward — extracts #### answer and compares."""
    solution_str = solution_str[-300:] if len(solution_str) > 300 else solution_str
    solutions = re.findall(r"#### (\-?[0-9\.,]+)", solution_str)
    if not solutions:
        # flexible fallback
        numbers = re.findall(r"(\-?[0-9\.,]+)", solution_str)
        final = None
        for n in reversed(numbers):
            if n not in ["", "."]:
                final = n
                break
    else:
        final = solutions[-1].replace(",", "").replace("$", "")

    if final is None:
        return 0.0
    # Clean both final and ground_truth consistently
    final = final.strip().replace(",", "").replace("$", "")
    try:
        gt = str(ground_truth).replace(",", "").replace("$", "").strip()
        return 1.0 if final == gt else 0.0
    except (ValueError, AttributeError):
        return 0.0


class RewardRegistry:
    """
    Pluggable reward registry for verl.
    Handles unknown data sources gracefully instead of raising NotImplementedError.
    """

    # Built-in handlers for data sources known to cause NotImplementedError
    _BUILTIN_HANDLERS: dict[str, Callable] = {
        "openai/gsm8k": _gsm8k_score,
    }

    def __init__(self, atropos_url: Optional[str] = None):
        self.atropos_url = atropos_url
        self._handlers: dict[str, Callable] = dict(self._BUILTIN_HANDLERS)

    def register(self, data_source: str, fn: Callable) -> None:
        """Register a custom reward function for a data source."""
        self._handlers[data_source] = fn
        logger.info(f"[RewardRegistry] Registered handler for {data_source}")

    def compute_score(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        **kwargs,
    ) -> float:
        """
        Compute reward score for a given data source.
        Falls back to registered handlers, then Atropos, then 0.0.
        """
        # Try verl's built-in first
        try:
            from verl.utils.reward_score import default_compute_score

            return default_compute_score(data_source, solution_str, ground_truth, **kwargs)
        except (NotImplementedError, ImportError, ModuleNotFoundError):
            pass
        except Exception as e:
            logger.warning(f"[RewardRegistry] verl built-in failed for {data_source}: {e}")

        # Try registered handler
        if data_source in self._handlers:
            try:
                score = self._handlers[data_source](solution_str, ground_truth)
                logger.debug(f"[RewardRegistry] Handler scored {data_source}: {score}")
                return score
            except Exception as e:
                logger.warning(f"[RewardRegistry] Handler failed for {data_source}: {e}")

        # Fall back to Atropos scoring if available
        if self.atropos_url:
            try:
                import requests

                resp = requests.post(
                    f"{self.atropos_url}/score",
                    json={
                        "solution": solution_str,
                        "ground_truth": ground_truth,
                        "data_source": data_source,
                    },
                    timeout=10,
                )
                score = resp.json().get("score", 0.0)
                logger.info(f"[RewardRegistry] Atropos scored {data_source}: {score}")
                return float(score)
            except Exception as e:
                logger.warning(f"[RewardRegistry] Atropos fallback failed: {e}")

        logger.error(f"[RewardRegistry] No handler found for {data_source}, returning 0.0")
        return 0.0


# Module-level default registry instance
_default_registry = RewardRegistry()


def register_reward(data_source: str, fn: Callable) -> None:
    """Register a reward function in the default registry."""
    _default_registry.register(data_source, fn)


def compute_score(data_source: str, solution_str: str, ground_truth: str, **kwargs) -> float:
    """Compute score using the default registry."""
    return _default_registry.compute_score(data_source, solution_str, ground_truth, **kwargs)
