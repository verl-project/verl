import os
import random
from typing import Any, Dict, List, Optional, Tuple

from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    Item,
    ScoredDataGroup,
)

# Attempt to import from helpers.py
from helpers import GPTPlayer, MaxDamagePlayer


class PokemonEnvConfig(BaseEnvConfig):
    """Configuration for the Pokemon Environment."""

    tokenizer_name: str = Field(
        default="cl100k_base", description="Tokenizer for GPT-4 models."
    )
    # total_steps here could mean total turns or total battles. Let's assume total battles for now.
    total_steps: int = Field(default=100, description="Total number of battles to run.")
    # Add other Pokemon-specific configurations here later
    battle_format: str = Field(
        default="gen9randombattle", description="Pokemon battle format."
    )


class PokemonEnv(BaseEnv):
    """
    An Atropos environment for training an agent to play Pokemon.
    This environment will use the poke-env library to simulate Pokemon battles.
    """

    name = "PokemonEnv"
    env_config_cls = PokemonEnvConfig

    def __init__(
        self,
        config: PokemonEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm=slurm, testing=testing)
        self.config: PokemonEnvConfig  # For type hinting

        self.server_configs = server_configs

        self.agent_player: Optional[MaxDamagePlayer] = None
        self.opponent_player: Optional[MaxDamagePlayer] = None
        self._active_battle_tag: Optional[str] = None
        self._current_available_moves: List[Move] = []
        self._current_available_switches: List[Pokemon] = []
        self._battles_played_count: int = 0

    @classmethod
    def config_init(cls) -> Tuple[PokemonEnvConfig, List[APIServerConfig]]:
        """Initializes the default configuration for the Pokemon environment."""
        env_config = PokemonEnvConfig(
            group_size=1,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=100,  # Number of battles
            batch_size=1,  # Batch size for Atropos, likely 1 item per trajectory step (one turn)
            steps_per_eval=10,
            max_token_length=4096,  # Increased for potentially long Pokemon battle prompts
            wandb_name="pokemon-env",
            battle_format="gen9randombattle",
            tokenizer_name="cl100k_base",  # For gpt-4 models
        )

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        server_configs = [
            APIServerConfig(
                base_url="https://api.openai.com/v1",
                model_name="gpt-4.1-nano",
                api_key=openai_api_key,
                num_requests_for_eval=10,  # Reduced for faster eval
            )
        ]
        # Removed print warning for missing API key as it was color-coded
        # and the framework or user should handle this check if critical.
        return env_config, server_configs

    async def setup(self):
        """
        Initial setup for the Pokemon environment.
        This is where you would initialize the poke-env player, connect to a showdown server (if needed),
        or load any other necessary resources.
        """
        print(f"[{self.name}] Setting up Pokemon environment...")

        self._battles_played_count = 0  # Ensure reset if setup is called multiple times

        llm_config = self.server_configs[0]
        self.agent_player = GPTPlayer(
            model_name=llm_config.model_name,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,  # Using rollout_server_url as the base_url
        )
        self.opponent_player = MaxDamagePlayer(battle_format=self.config.battle_format)

        print(f"[{self.name}] Pokemon environment setup complete.")

    async def get_next_item(self) -> Optional[Item]:
        # Pokemon battles are random, so we just return a random number
        return random.randint(0, 10000)

    async def collect_trajectories(
        self,
        item: Item,  # Item is not used in this specific implementation but is part of the signature
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        if not self.agent_player or not self.opponent_player:
            print(f"[{self.name}] Players not initialized for collect_trajectories.")
            return None, []

        num_battles_in_group = self.config.group_size
        if num_battles_in_group <= 0:
            # If group_size is zero or negative, no battles to play.
            # Return an empty ScoredDataGroup or None as appropriate.
            # According to Atropos, usually None is for "pause worker", so empty group might be better if 0 is valid.
            # For simplicity, if group_size is 0, we produce nothing.
            return None, []

        existing_battle_tags = set(self.agent_player.battles.keys())

        try:
            await self.agent_player.battle_against(
                self.opponent_player, n_battles=num_battles_in_group
            )
        except Exception as e:
            print(
                f"[{self.name}] Error during batch battle_against for {num_battles_in_group} battles: {e}"
            )
            # If the batch call itself fails catastrophically, we might not have reliable partial data.
            # For now, returning None, indicating an issue in collecting the group.
            return None, []

        current_battle_tags = set(self.agent_player.battles.keys())
        newly_added_battle_tags_list = list(current_battle_tags - existing_battle_tags)

        # Check if the number of newly found battles matches the expected group size.
        # `battle_against` might play fewer than n_battles if errors occur internally for some battles.
        # For Atropos, we generally need a full group or handle deviations explicitly.
        # If poke-env played fewer than requested, we have a mismatch.
        if len(newly_added_battle_tags_list) != num_battles_in_group:
            print(
                f"[{self.name}] Warning/Error: Expected {num_battles_in_group} new battles, "
                f"but identified {len(newly_added_battle_tags_list)} new battle tags."
            )
            print(f"[{self.name}] Identified tags: {newly_added_battle_tags_list}")
            # Decide how to handle: return None, try to fill, or return partial (latter is hard for Atropos)
            # For now, strict check: if not all battles were played and recorded as expected, this is an issue.
            return None, []

        raw_prompts_group: List[str] = []
        raw_completions_group: List[str] = []
        prompt_tokens_group: List[List[int]] = []
        completion_tokens_group: List[List[int]] = []
        rewards_group: List[List[float]] = []
        meta_data_group: List[Dict[str, Any]] = []

        initial_battles_played_this_session = self._battles_played_count

        for i, battle_tag in enumerate(newly_added_battle_tags_list):
            battle_object = self.agent_player.battles.get(battle_tag)

            # This is the unique, incrementing number for each battle processed by this env instance
            current_battle_overall_seq_num = initial_battles_played_this_session + i + 1

            outcome_str = "Unknown"
            reward_value = 0.0

            if not battle_object:
                print(
                    f"[{self.name}] Error: Battle object for presumed new tag {battle_tag} "
                    f"not found in player.battles."
                )
                outcome_str = "Error: Battle data missing"
                # reward_value remains 0.0
            else:
                if battle_object.won is True:
                    outcome_str = "Won"
                    reward_value = 1.0
                elif battle_object.won is False:
                    outcome_str = "Lost"
                    reward_value = 0.0
                else:  # battle_object.won is None (can mean tie or battle didn't properly conclude with win/loss)
                    outcome_str = "Draw/Unfinished"
                    reward_value = 0.0

            print(
                f"[{self.name}] Processed Battle (Overall #{current_battle_overall_seq_num}, "
                f"Group Item {i+1}/{num_battles_in_group}, Tag: {battle_tag}). "
                f"Agent outcome: {outcome_str}"
            )

            raw_prompt = (
                f"Battle {current_battle_overall_seq_num}: Agent vs Opponent "
                f"({self.config.battle_format}, Tag: {battle_tag})"
            )
            raw_completion = f"Outcome: {outcome_str}"
            prompt_tokens = self.tokenizer.encode(raw_prompt)
            completion_tokens = self.tokenizer.encode(raw_completion)

            raw_prompts_group.append(raw_prompt)
            raw_completions_group.append(raw_completion)
            prompt_tokens_group.append(prompt_tokens)
            completion_tokens_group.append(completion_tokens)
            rewards_group.append(
                [reward_value]
            )  # Reward is a list containing a single float
            meta_data_group.append(
                {
                    "battle_sequence_number": current_battle_overall_seq_num,
                    "battle_tag": battle_tag,
                    "outcome": outcome_str,
                    "agent_total_wins_after_batch": self.agent_player.n_won_battles,  # Snapshot after the whole batch
                    "battles_played_this_session_after_item": current_battle_overall_seq_num,
                }
            )

        # Crucially, update the total battles played count *after* processing the entire group
        self._battles_played_count += len(newly_added_battle_tags_list)

        if (
            not raw_prompts_group
        ):  # If somehow the loop didn't run (e.g. num_battles_in_group was 0 but check missed)
            return None, []

        group_data = ScoredDataGroup(
            messages=raw_prompts_group,
            tokens=prompt_tokens_group,
            masks=completion_tokens_group,
            scores=rewards_group,
            meta_data=meta_data_group,
        )

        return group_data, []

    async def evaluate(self, *args, **kwargs):
        """
        Perform evaluation runs for the Pokemon environment.
        This could involve playing a set number of games against a fixed opponent
        or evaluating performance on specific battle scenarios.
        Log metrics like win rate, average damage, etc.
        """
        print(f"[{self.name}] Starting evaluation...")

        num_battles_total = (
            self.agent_player.n_finished_battles if self.agent_player else 0
        )
        # Use the player's win_rate attribute directly
        win_rate = (
            self.agent_player.win_rate
            if self.agent_player and num_battles_total > 0
            else 0.0
        )

        eval_metrics = {
            "eval_win_rate": win_rate,
            "eval_battles_played": num_battles_total,
            "eval_battles_won": (
                self.agent_player.n_won_battles if self.agent_player else 0
            ),
        }
        print(f"[{self.name}] Evaluation complete. Metrics: {eval_metrics}")

        if self.config.use_wandb and self.wandb_run:
            self.wandb_run.log(
                eval_metrics, step=self.curr_step
            )  # curr_step is Atropos step

    async def cleanup(self):
        print(f"[{self.name}] Cleaning up Pokemon environment...")
        # if self._battle_task and not self._battle_task.done(): # Removed battle_task
        #     self._battle_task.cancel()
        #     try:
        #         await self._battle_task
        #     except asyncio.CancelledError:
        #         print(f"[{self.name}] Battle task cancelled.")
        # # Close queues by sending sentinel value if not already done # Queues removed
        # if self._observation_queue and hasattr(self._observation_queue, "put_nowait"): # Queues removed
        #     try:
        #         self._observation_queue.put_nowait(None)
        #     except asyncio.QueueFull:
        #         pass
        # if self._action_queue and hasattr(self._action_queue, "put_nowait"): # Queues removed
        #     try:
        #         self._action_queue.put_nowait(None)  # To unblock player if waiting
        #     except asyncio.QueueFull:
        #         pass
        # Ensure agent and opponent players are cleaned up if they have explicit cleanup methods
        # poke-env players (like MaxDamagePlayer) typically don't require explicit async cleanup
        # unless they are managing websocket connections for online play, which MaxDamagePlayer doesn't by default.

        await super().cleanup()
        print(f"[{self.name}] Pokemon environment cleanup complete.")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add metrics specific to the Pokemon environment based on overall agent stats
        if self.agent_player:
            wandb_metrics["battles_finished_total"] = (
                self.agent_player.n_finished_battles
            )
            wandb_metrics["battles_won_total"] = self.agent_player.n_won_battles
            if self.agent_player.n_finished_battles > 0:
                wandb_metrics["win_rate_overall"] = self.agent_player.win_rate
            else:
                wandb_metrics["win_rate_overall"] = 0.0

        # Calculate metrics from the rollouts completed in the current logging period
        # self.completed_rollouts is a deque of ScoredDataGroup objects from BaseEnv
        period_total_reward = 0.0
        period_total_battles = 0
        if hasattr(self, "completed_rollouts") and self.completed_rollouts:
            for group_data in self.completed_rollouts:
                if group_data and hasattr(group_data, "rewards") and group_data.rewards:
                    for battle_reward_list in group_data.rewards:
                        if battle_reward_list:  # Should be a list like [1.0] or [0.0]
                            period_total_reward += battle_reward_list[0]
                        period_total_battles += (
                            1  # Count each entry in group_data.rewards as a battle
                        )

        if period_total_battles > 0:
            wandb_metrics["avg_reward_in_period"] = (
                period_total_reward / period_total_battles
            )
        else:
            wandb_metrics["avg_reward_in_period"] = 0.0  # Or None, or omit
        wandb_metrics["battles_in_period"] = period_total_battles

        # Call the superclass wandb_log to handle logging of base metrics and actual rollouts (as W&B Tables)
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    PokemonEnv.cli()
