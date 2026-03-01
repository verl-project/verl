# reward_fns/bet_sizing.py
import re
from typing import Any, List, Optional

from atroposlib.envs.reward_fns import RewardFunction, registry


@registry.register
class PokerBetSizingReward(RewardFunction):
    """Reward function specifically for evaluating bet sizing accuracy"""

    def __init__(
        self,
        perfect_match_score: float = 1.0,
        min_score: float = 0.0,
        max_deviation_pct: float = 0.5,  # Max deviation for non-zero score
        weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(weight=weight, **kwargs)
        self.perfect_match_score = perfect_match_score
        self.min_score = min_score
        self.max_deviation_pct = max_deviation_pct

    def compute(
        self, completions: List[Any], winner_action: Optional[str] = None, **kwargs
    ) -> List[float]:
        """Score bet sizing accuracy compared to winner's action"""
        if winner_action is None:
            return [0.0] * len(completions)

        # Extract winner bet amount if present
        winner_parts = winner_action.lower().split()
        if len(winner_parts) < 2 or winner_parts[0] not in ["bet", "raise"]:
            return [0.0] * len(completions)  # Not a betting action

        try:
            winner_amount = float(winner_parts[1])
        except (ValueError, IndexError):
            return [0.0] * len(completions)  # Invalid amount format

        scores = []
        for completion in completions:
            content = self.get_content(completion).lower()

            # Check if it's a betting action
            if "bet" not in content and "raise" not in content:
                scores.append(0.0)
                continue

            # Extract amount
            amount_matches = re.findall(r"(\d+(?:\.\d+)?)", content)
            if not amount_matches:
                scores.append(0.0)
                continue

            response_amount = float(amount_matches[0])

            # Calculate deviation
            max_amount = max(response_amount, winner_amount)
            min_amount = min(response_amount, winner_amount)

            if min_amount == 0:
                deviation = 1.0  # Maximum deviation
            else:
                deviation = (max_amount - min_amount) / min_amount

            # Score based on deviation
            if deviation > self.max_deviation_pct:
                score = self.min_score
            else:
                # Linear interpolation between perfect and min score
                score = self.perfect_match_score - (
                    deviation / self.max_deviation_pct
                ) * (self.perfect_match_score - self.min_score)

            scores.append(score)

        return scores
