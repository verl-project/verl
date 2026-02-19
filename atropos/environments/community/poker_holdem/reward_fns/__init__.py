# Import custom reward functions to register them
from atroposlib.envs.reward_fns import CombinedReward

from .action_match import PokerActionMatchReward
from .bet_sizing import PokerBetSizingReward


class PokerCombinedReward(CombinedReward):
    """Pre-configured combined reward for poker action evaluation"""

    def __init__(
        self,
        action_match_weight: float = 0.6,
        bet_sizing_weight: float = 0.4,
        normalization: str = "sum",
        weight: float = 1.0,
    ):

        self.normalization = normalization
        self.reward_functions = []
        self.weight = weight

        # Initialize all sub-reward functions
        self.reward_functions.append(PokerActionMatchReward(weight=action_match_weight))
        self.reward_functions.append(PokerBetSizingReward(weight=bet_sizing_weight))
