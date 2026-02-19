import asyncio
import logging
import os
import random

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum
from environments.game_environments.gymnasium.blackjack.blackjack_env_thinking import (
    BlackjackEnv,
    BlackjackEnvConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting Blackjack environment local debug runner")

    env_config = BlackjackEnvConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=1,
        use_wandb=False,
        wandb_name="blackjack_local_debug",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        max_token_length=1024 * 4,
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        env_name="Blackjack-v1",
        temperature=0.2,
        top_p=0.9,
        max_turns=5,
        thinking_active=True,
        eval_episodes=0,
        max_think_chars_history=3000,
        max_trajectory_tokens=24576,
        debug_mode=True,
        mc_samples=1,
    )
    server_configs = [
        APIServerConfig(
            model_name="gpt-4.1-nano",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        )
    ]
    logger.info("Using hardcoded debug configuration.")
    logger.debug(f"Env Config: {env_config}")
    logger.debug(f"Server Configs: {server_configs}")

    try:
        env = BlackjackEnv(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize BlackjackEnv: {e}")
        return

    logger.info("Running a single trajectory directly")
    try:
        await env.setup()
        seed = random.randint(0, 1000000)
        logger.info(f"Using seed: {seed}")

        _ = env._get_or_create_episode(seed)

        result_trajectories_tuple = await env.collect_trajectories((seed, 0))
        result_trajectory = result_trajectories_tuple[0]

        logger.info(
            f"Trajectory collection complete with {len(result_trajectory)} groups/steps."
        )

        episode_summary = None
        if env.completed_episode_metrics_buffer:
            episode_summary = env.completed_episode_metrics_buffer[-1]

        if episode_summary and episode_summary.get("seed") == seed:
            logger.info("\n========== Episode Summary ==========")
            logger.info(f"Seed: {episode_summary['seed']}")
            logger.info(f"Total steps taken: {episode_summary['num_steps']}")
            logger.info(
                f"Final Environment reward: {episode_summary['total_reward']:.2f}"
            )

            game_outcome_val = episode_summary.get("game_outcome", 0)
            outcome_str = "Draw"
            if game_outcome_val == 1:
                outcome_str = "Win"
            elif game_outcome_val == -1:
                outcome_str = "Loss"
            logger.info(
                f"Game Outcome: {outcome_str} (Reward: {episode_summary['total_reward']:.0f})"
            )

            if episode_summary["num_total_actions"] > 0:
                accuracy = episode_summary["num_correct_actions"] / max(
                    1, episode_summary["num_total_actions"]
                )
                logger.info(
                    f"Action accuracy (valid tool calls): "
                    f"{episode_summary['num_correct_actions']}/{episode_summary['num_total_actions']} "
                    f"({accuracy:.2%})"
                )
            else:
                logger.info(
                    "Action accuracy (valid tool calls): No tool calls attempted or recorded."
                )
            logger.info("=======================================")
        else:
            logger.error(
                f"Could not get episode summary for seed {seed} from metrics buffer or seed mismatch."
            )

    except Exception as e:
        logger.exception(
            f"An error occurred during trajectory collection or summary: {e}"
        )


if __name__ == "__main__":
    asyncio.run(main())
