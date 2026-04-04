#!/usr/bin/env python3
import asyncio
import logging
import os

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig
from environments.infinimath.infinimath_env import (
    InfiniteMathEnv,
    InfiniteMathEnvConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting InfiniteMath environment local runner")

    config = InfiniteMathEnvConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=1,
        use_wandb=False,
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        max_token_length=2048,
        wandb_name="infinite_math_local_debug",
        ensure_scores_are_not_same=False,
        starting_level=1,
        progress_threshold=0.8,
        min_evaluations=3,
        correct_reward=1.0,
        incorrect_reward=-0.5,
        think_block_bonus=0.1,
        boxed_answer_bonus=0.2,
        apply_length_penalty=False,
        length_threshold_ratio=0.6,
        temperature=0.3,
        top_p=0.9,
        word_problem_model_name="gpt-4.1-mini",
        word_problem_openai_api_key=os.getenv("OPENAI_API_KEY_WORD_PROBLEM")
        or os.getenv("OPENAI_API_KEY"),
        word_problem_openai_base_url=os.getenv("OPENAI_BASE_URL_WORD_PROBLEM"),
    )

    server_configs = [
        APIServerConfig(
            model_name="gpt-4.1-nano",
            base_url=None,
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=600,
        )
    ]

    logger.info("Using hardcoded debug configuration.")
    logger.debug(f"Env Config: {config}")
    logger.debug(f"Server Configs: {server_configs}")

    try:
        env = InfiniteMathEnv(
            config=config,
            server_configs=server_configs,
            slurm=False,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize InfiniteMathEnv: {e}")
        return

    logger.info("Setting up environment...")
    await env.setup()
    logger.info("Environment setup complete.")

    logger.info("Getting a math problem...")
    item = await env.get_next_item()
    problem_prompt, solution, generator_id = item

    problem_content = dict(problem_prompt[0])["content"]
    logger.info(
        f"Problem (ID: {generator_id}, Level: {env.curriculum.get_current_level()}): {problem_content}"
    )
    logger.info(f"Expected Solution: {solution}")

    logger.info("Collecting trajectories...")
    trajectories_data, backlog = await env.collect_trajectories(item)

    if not trajectories_data:
        logger.error("No trajectories were collected.")
        return

    logger.info(
        f"Collected {len(trajectories_data)} data points for scoring (should be 1 for group_size=1)."
    )

    logger.info("Scoring trajectories...")
    scored_data = await env.score(trajectories_data)

    logger.info("\n========== Trajectory Summary ==========")
    if scored_data and scored_data.get("messages") and scored_data.get("scores"):
        for i, messages_list in enumerate(scored_data["messages"]):
            assistant_response = ""
            if messages_list and messages_list[-1].get("role") == "assistant":
                assistant_response = messages_list[-1].get("content", "N/A")

            logger.info(f"--- Attempt {i+1} ---")
            logger.info(f"Problem: {problem_content}")
            logger.info(f"Full Assistant Response:\\n{assistant_response}")
            logger.info(f"Score: {scored_data['scores'][i]}")
            is_correct_task = env.check_answer(assistant_response, solution)
            logger.info(f"Checked Correct by env.check_answer: {is_correct_task}")

        correct_count_buffer = sum(env.percent_correct_buffer)
        total_attempts_buffer = len(env.percent_correct_buffer)

        logger.info("\n--- Overall for this run ---")
        logger.info(f"Expected Solution: {solution}")
        logger.info(f"Score(s) from env.score: {scored_data['scores']}")
        if total_attempts_buffer > 0:
            logger.info(
                f"Correct based on internal buffer: {correct_count_buffer}/{total_attempts_buffer}"
            )
        else:
            logger.info("No attempts recorded in percent_correct_buffer.")

    else:
        logger.error("Scored data is missing expected fields ('messages' or 'scores').")

    logger.info("=======================================")

    # Re-add curriculum and evaluation testing
    logger.info("\n=== Testing Evaluation Function ===")

    # Record the current level
    initial_level_eval = env.curriculum.get_current_level()
    logger.info(f"Current level before evaluation: {initial_level_eval}")
    logger.info(f"Level description: {env.curriculum.get_level_description()}")
    logger.info(f"Progress threshold: {env.curriculum.progress_threshold}")
    logger.info(f"Min evaluations needed: {env.curriculum.min_evaluations}")

    # Run the evaluate method
    eval_metrics = await env.evaluate()

    # Display evaluation results
    logger.info("Evaluation metrics:")
    if eval_metrics:
        for metric_name, metric_value in eval_metrics:
            logger.info(f"  - {metric_name}: {metric_value}")
    else:
        logger.info("  No evaluation metrics returned.")

    # Check if the level advanced
    new_level_eval = env.curriculum.get_current_level()
    if new_level_eval > initial_level_eval:
        logger.info(
            f"Successfully advanced from level {initial_level_eval} to level {new_level_eval} during evaluation!"
        )
        logger.info(f"New level description: {env.curriculum.get_level_description()}")
    else:
        logger.info(
            f"Did not advance during evaluation. Remained at level {initial_level_eval}."
        )
        # Show current progress toward advancement
        current_level_desc = env.curriculum.get_current_level()
        if current_level_desc in env.curriculum.performance_history:
            history = env.curriculum.performance_history[current_level_desc]
            if len(history) >= env.curriculum.min_evaluations:
                recent_history = history[-env.curriculum.min_evaluations :]
                success_rate = sum(recent_history) / len(recent_history)
                logger.info(
                    f"Current success rate for level {current_level_desc}: {success_rate:.2f} "
                    f"(need {env.curriculum.progress_threshold} to advance)"
                )
            else:
                logger.info(
                    f"Need more evaluations for level {current_level_desc}: "
                    f"{len(history)}/{env.curriculum.min_evaluations}"
                )

    # Show all levels and their performance history after evaluation
    logger.info("\nPerformance history by level (after evaluation run):")
    for level_hist_key in sorted(env.curriculum.performance_history.keys()):
        history_list = env.curriculum.performance_history[level_hist_key]
        if history_list:
            success_rate_hist = sum(history_list) / len(history_list)
            logger.info(
                f"  Level {level_hist_key}: {success_rate_hist:.2f} ({sum(history_list)}/{len(history_list)} correct)"
            )
        else:
            logger.info(f"  Level {level_hist_key}: No data")

    # Test curriculum advancement with simulated performance history
    logger.info("\n=== Testing Curriculum Advancement Manually ===")
    initial_level_manual_adv = env.curriculum.get_current_level()
    logger.info(
        f"Starting manual advancement test from level: {initial_level_manual_adv}"
    )

    # Simulate good performance at current level
    # Ensure we don't try to get items if curriculum is already at max level from previous eval
    max_level_possible = max(env.curriculum.DIFFICULTY_LEVELS.keys())
    if initial_level_manual_adv < max_level_possible:
        logger.info(
            f"Simulating {config.min_evaluations} correct answers for level {initial_level_manual_adv}..."
        )
        for _ in range(config.min_evaluations):  # Use config for min_evaluations
            # Get a problem from current level to ensure generator_id is valid for the level
            # The level might have changed due to the previous env.evaluate() call
            problem_item_adv_test = await env.get_next_item()
            _, _, generator_id_adv_test = problem_item_adv_test
            env.curriculum.record_performance(generator_id_adv_test, True)

        # Try to advance difficulty
        did_advance = env.curriculum.advance_difficulty()
        new_level_manual_adv = env.curriculum.get_current_level()

        logger.info("Curriculum advancement test results:")
        logger.info(f"  - Level before manual simulation: {initial_level_manual_adv}")
        logger.info(f"  - Recorded {config.min_evaluations} correct answers manually.")
        logger.info(f"  - Did advance: {did_advance}")
        logger.info(
            f"  - Level after manual advancement attempt: {new_level_manual_adv}"
        )
    else:
        logger.info(
            "Skipping manual advancement simulation as current level "
            f"{initial_level_manual_adv} is already max level {max_level_possible}."
        )

    logger.info("InfiniteMath local runner completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
