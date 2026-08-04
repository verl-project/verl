import json
import logging
import random
from typing import Dict, List, Optional, Tuple

import gymnasium as gym

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)

ACTION_HIT = 1
ACTION_STICK = 0
ACTION_MAP_TO_STR = {ACTION_HIT: "hit", ACTION_STICK: "stick"}
ACTION_STR_TO_INT = {v: k for k, v in ACTION_MAP_TO_STR.items()}


class BlackjackEnvNoThinkingConfig(BaseEnvConfig):
    """
    Configuration for the BlackjackEnvNoThinking environment.
    """

    env_name: str = "Blackjack-v1"
    max_episode_turns: int = 10
    eval_episodes: int = 100


class BlackjackEnvNoThinking(BaseEnv):
    name = "blackjack_no_thinking"
    env_config_cls = BlackjackEnvNoThinkingConfig

    def __init__(
        self,
        config: BlackjackEnvNoThinkingConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: BlackjackEnvNoThinkingConfig = config
        self.episode_outcomes_buffer: List[float] = []
        self.eval_metrics_custom: List[Tuple[str, float]] = []

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "take_action",
                    "description": "Choose to 'hit' or 'stick' in Blackjack.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["hit", "stick"]}
                        },
                        "required": ["action"],
                    },
                },
            }
        ]

        tools_json = json.dumps(self.tools)
        self.system_prompt = (
            "You are an AI agent playing Blackjack. "
            "You need to decide whether to hit or stick based on your current hand and the dealer's showing card.\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            '<tool_call>\n{"arguments": {"action": "hit"}, "name": "take_action"}\n</tool_call>\n\n'
            "Your full answer format should be (NO THINKING BLOCK):\n"
            '<tool_call>\n{"arguments": {"action": "stick"}, "name": "take_action"}\n</tool_call>\n'
        )

    @classmethod
    def config_init(cls) -> Tuple[BlackjackEnvNoThinkingConfig, List[APIServerConfig]]:
        env_config = BlackjackEnvNoThinkingConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=2048,
            wandb_name=cls.name,
            steps_per_eval=50,
            max_episode_turns=10,
            eval_episodes=100,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
        ]
        return env_config, server_configs

    def _format_observation(self, obs: Tuple[int, int, int]) -> str:
        """Converts a Blackjack observation to a human-readable string."""
        player_sum, dealer_card, usable_ace = obs
        return (
            f"Your current hand sum is {player_sum}. "
            f"The dealer is showing a {dealer_card}. "
            f"You have a usable ace: {'yes' if usable_ace else 'no'}."
        )

    def _parse_action_from_llm(self, llm_response: str) -> Optional[int]:
        """Parses the action from the LLM's tool_call response."""
        if not llm_response:
            logger.warning(
                "Attempted to parse an empty LLM response. Returning invalid action (None)."
            )
            return None

        parsed_name, parsed_args, is_error = parse_tool_call(
            llm_response, self.tools, ["tool_call"]
        )

        if is_error:
            error_detail = (
                str(parsed_name)
                if parsed_name
                else "Parser indicated error, but no specific message was returned."
            )
            logger.warning(
                f"Failed to parse tool call. Full response: '{llm_response}'. Error: {error_detail}"
            )
            return None

        if parsed_name != "take_action":
            logger.warning(
                f"Expected tool call name 'take_action', but got '{parsed_name}'. Response: '{llm_response}'"
            )
            return None

        action_str = parsed_args.get("action", "").lower()
        if action_str == "hit":
            return ACTION_HIT
        elif action_str == "stick":
            return ACTION_STICK
        else:
            logger.warning(
                f"Successfully parsed tool call '{parsed_name}', "
                f"but action argument is invalid. Action: '{action_str}'. "
                f"Full response: '{llm_response}'. Parsed args: {parsed_args}"
            )
            return None

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Collects a single trajectory (episode) for the Blackjack environment.
        The LLM directly outputs 'hit' or 'stick'.
        The 'score' in ScoredDataItem is the final game outcome (+1, 0, -1).
        """
        seed = item["seed"]
        messages: List[Message] = []
        game_reward = 0.0

        try:
            env = gym.make(self.config.env_name)
        except Exception as e:
            logger.error(f"Failed to make environment {self.config.env_name}: {e}")
            return None, []

        try:
            obs, info = env.reset(seed=seed)
        except Exception as e:
            logger.error(f"Failed to reset environment with seed {seed}: {e}")
            env.close()
            return None, []

        messages.append({"role": "system", "content": self.system_prompt})

        current_obs_str = self._format_observation(obs)
        messages.append({"role": "user", "content": current_obs_str})

        async with self.server.dedicated_server() as server:
            for _ in range(self.config.max_episode_turns):
                if (
                    len(self.tokenizer.apply_chat_template(messages, tokenize=False))
                    > self.config.max_token_length - 50
                ):
                    logger.warning(
                        f"[Seed: {seed}] Max token length reached, truncating episode."
                    )
                    break

                max_tokens_for_action = 512

                try:
                    chat_completions = await server.chat_completion(
                        messages=messages,
                        n=1,
                        max_tokens=max_tokens_for_action,
                        temperature=0.5,
                    )
                    llm_action_response = chat_completions.choices[
                        0
                    ].message.content.strip()
                    logger.info(
                        f"[Seed: {seed}] LLM Raw Response: '{llm_action_response}'"
                    )
                except Exception as e:
                    logger.error(f"[Seed: {seed}] LLM API error: {e}")
                    break

                messages.append({"role": "assistant", "content": llm_action_response})

                action = self._parse_action_from_llm(llm_action_response)
                if action is None:
                    logger.warning(
                        f"[Seed: {seed}] Invalid action parsed. Ending episode."
                    )
                    game_reward = -1.0
                    break

                try:
                    obs, reward, terminated, truncated, _ = env.step(action)
                    game_reward = float(reward)
                except Exception as e:
                    logger.error(f"[Seed: {seed}] Error stepping env: {e}")
                    break

                if terminated or truncated:
                    break

                current_obs_str = self._format_observation(obs)
                messages.append({"role": "user", "content": current_obs_str})

        env.close()
        self.episode_outcomes_buffer.append(game_reward)

        tokenization_result = tokenize_for_trainer(
            tokenizer=self.tokenizer, chat=messages, train_on_all_assistant_turns=True
        )

        tokens = tokenization_result["tokens"]
        masks = tokenization_result["masks"]

        scored_data_item = ScoredDataItem(
            messages=messages if self.config.include_messages else None,
            tokens=tokens,
            masks=masks,
            scores=game_reward,
        )
        return scored_data_item, []

    async def get_next_item(self) -> Item:
        next_seed = random.randint(0, 1_000_000)
        return {"seed": next_seed}

    async def setup(self):
        logger.info(f"Setting up {self.name} environment.")

    async def evaluate(self, *args, **kwargs):
        logger.info(
            f"Starting evaluation for {self.name} with {self.config.eval_episodes} episodes."
        )

        wins = 0
        losses = 0
        draws = 0

        eval_outcomes: List[float] = []

        for i in range(self.config.eval_episodes):
            seed = random.randint(1_000_001, 2_000_000)
            item = {"seed": seed}
            scored_item_tuple = await self.collect_trajectory(item)
            if scored_item_tuple and scored_item_tuple[0]:
                outcome = scored_item_tuple[0]["scores"]
                eval_outcomes.append(outcome)
            else:
                logger.warning(
                    f"Evaluation episode {i+1} (seed {seed}) failed to produce data."
                )

        if not eval_outcomes:
            logger.warning("No evaluation episodes completed successfully.")
            self.eval_metrics_custom = []
            return

        for outcome in eval_outcomes:
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1

        num_completed = len(eval_outcomes)
        win_rate = wins / num_completed if num_completed > 0 else 0
        loss_rate = losses / num_completed if num_completed > 0 else 0
        draw_rate = draws / num_completed if num_completed > 0 else 0
        avg_reward = sum(eval_outcomes) / num_completed if num_completed > 0 else 0

        self.eval_metrics_custom = [
            (f"{self.name}_eval/win_rate", win_rate),
            (f"{self.name}_eval/loss_rate", loss_rate),
            (f"{self.name}_eval/draw_rate", draw_rate),
            (f"{self.name}_eval/avg_reward", avg_reward),
            (f"{self.name}_eval/num_completed_episodes", num_completed),
        ]
        logger.info(
            f"Evaluation completed for {self.name}. Metrics: {self.eval_metrics_custom}"
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.episode_outcomes_buffer:
            avg_training_reward = sum(self.episode_outcomes_buffer) / len(
                self.episode_outcomes_buffer
            )
            wandb_metrics[f"{self.name}_train/avg_episode_reward"] = avg_training_reward
            train_wins = sum(1 for r in self.episode_outcomes_buffer if r > 0)
            train_losses = sum(1 for r in self.episode_outcomes_buffer if r < 0)
            train_draws = sum(1 for r in self.episode_outcomes_buffer if r == 0)
            wandb_metrics[f"{self.name}_train/win_count"] = train_wins
            wandb_metrics[f"{self.name}_train/loss_count"] = train_losses
            wandb_metrics[f"{self.name}_train/draw_count"] = train_draws
            wandb_metrics[f"{self.name}_train/num_episodes_in_batch"] = len(
                self.episode_outcomes_buffer
            )

        self.episode_outcomes_buffer = []

        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    BlackjackEnvNoThinking.cli()
