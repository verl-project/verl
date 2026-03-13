# reward_fns/action_match.py
import re
from typing import Any, List, Optional

from atroposlib.envs.reward_fns import RewardFunction, registry


@registry.register
class PokerActionMatchReward(RewardFunction):
    """Reward function that scores based on match to winning player's action"""

    def __init__(
        self,
        exact_match_score: float = 1.0,
        action_type_score: float = 0.7,
        related_action_score: float = 0.5,
        weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(weight=weight, **kwargs)
        self.exact_match_score = exact_match_score
        self.action_type_score = action_type_score
        self.related_action_score = related_action_score

    def compute(
        self, completions: List[Any], winner_action: Optional[str] = None, **kwargs
    ) -> List[float]:
        """Score completions based on similarity to winner's action"""
        if winner_action is None:
            return [0.0] * len(completions)

        scores = []
        for completion in completions:
            content = self.get_content(completion)
            scores.append(self._score_single_response(content, winner_action))

        return scores

    def _score_single_response(self, response: str, winner_action: str) -> float:
        """Score a single response based on similarity to winner's action"""
        if winner_action is None:
            return 0.0  # No winner action to compare against

        # Extract action type and amount if present
        winner_parts = winner_action.lower().split()
        winner_action_type = winner_parts[0]
        winner_amount = float(winner_parts[1]) if len(winner_parts) > 1 else None

        # Look for action keywords in response
        response = response.lower()

        # Check for exact action match
        if winner_action_type in response:
            # Perfect match on action type
            if winner_amount is None:
                return (
                    self.exact_match_score
                )  # Full points for matching action with no amount

            # Look for amount in response
            amount_matches = re.findall(r"(\d+(?:\.\d+)?)", response)
            if amount_matches:
                response_amount = float(amount_matches[0])
                # Score based on how close the amount is
                amount_ratio = min(response_amount, winner_amount) / max(
                    response_amount, winner_amount
                )
                return self.action_type_score + (
                    (self.exact_match_score - self.action_type_score) * amount_ratio
                )

            return self.action_type_score  # Matched action but no amount

        # Check for related actions (partial credit)
        aggressive_actions = ["bet", "raise"]
        passive_actions = ["check", "call"]
        fold_action = "fold"

        if winner_action_type in aggressive_actions and any(
            a in response for a in aggressive_actions
        ):
            return (
                self.related_action_score
            )  # Partial credit for similar aggressive action
        elif winner_action_type in passive_actions and any(
            a in response for a in passive_actions
        ):
            return (
                self.related_action_score
            )  # Partial credit for similar passive action
        elif winner_action_type == fold_action and fold_action in response:
            return self.exact_match_score  # Full credit for fold (binary decision)

        return 0.0  # No match
