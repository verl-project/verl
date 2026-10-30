import asyncio
import logging
import os
import random
from typing import Optional

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum, ScoredDataItem
from environments.game_environments.gymnasium.blackjack.blackjack_env_no_thinking import (
    BlackjackEnvNoThinking,
    BlackjackEnvNoThinkingConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting Blackjack (No Thinking) environment local debug runner")

    env_config = BlackjackEnvNoThinkingConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=1,
        use_wandb=False,
        wandb_name="blackjack_no_thinking_local_debug",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        max_token_length=1024,
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        env_name="Blackjack-v1",
        max_episode_turns=10,
        eval_episodes=0,
    )
    server_configs = [
        APIServerConfig(
            model_name="gpt-4.1-nano",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        )
    ]
    logger.info("Using hardcoded debug configuration for No Thinking Blackjack.")
    logger.debug(f"Env Config: {env_config}")
    logger.debug(f"Server Configs: {server_configs}")

    try:
        env = BlackjackEnvNoThinking(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=False,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize BlackjackEnvNoThinking: {e}")
        return

    logger.info("Running a single trajectory directly using collect_trajectory")
    try:
        await env.setup()
        seed = random.randint(0, 1000000)
        item_for_env = {"seed": seed}
        logger.info(f"Using seed: {seed} for item: {item_for_env}")

        result_tuple = await env.collect_trajectory(item_for_env)

        scored_data_item: Optional[ScoredDataItem] = None
        if result_tuple and result_tuple[0]:
            scored_data_item = result_tuple[0]
            logger.info(
                f"Trajectory collection complete. Score: {scored_data_item.get('scores')}"
            )
            if env_config.include_messages and scored_data_item.get("messages"):
                logger.info("Collected Messages:")
                for i, msg in enumerate(scored_data_item["messages"]):
                    logger.info(
                        f"  {i}. Role: {msg['role']}, Content: '{str(msg['content'])[:150]}...'"
                    )
            logger.info(
                f"Tokens ({len(scored_data_item.get('tokens', []))}): {str(scored_data_item.get('tokens'))[:100]}..."
            )
            logger.info(
                f"Masks ({len(scored_data_item.get('masks', []))}): {str(scored_data_item.get('masks'))[:100]}..."
            )
        else:
            logger.error("Trajectory collection did not return a ScoredDataItem.")

        episode_summary_reward = None
        if env.episode_outcomes_buffer:
            episode_summary_reward = env.episode_outcomes_buffer[-1]

        if episode_summary_reward is not None:
            logger.info("\n========== Episode Summary ==========")
            logger.info(f"Seed: {seed}")
            logger.info(
                f"Final Environment reward (Score): {episode_summary_reward:.2f}"
            )
            outcome_str = "Draw"
            if episode_summary_reward > 0:
                outcome_str = "Win"
            elif episode_summary_reward < 0:
                outcome_str = "Loss"
            logger.info(f"Game Outcome: {outcome_str}")
            logger.info("=======================================")
        else:
            logger.error(
                f"Could not get episode summary for seed {seed} from metrics buffer."
            )

    except Exception as e:
        logger.exception(
            f"An error occurred during trajectory collection or summary: {e}"
        )


if __name__ == "__main__":
    asyncio.run(main())
