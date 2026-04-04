#!/usr/bin/env python3
"""
Local testing script for InternBootcamp environment with RandomTask
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum
from environments.intern_bootcamp.intern_bootcamp_env import (
    InternBootcampEnv,
    InternBootcampEnvConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting InternBootcamp environment local test runner with RandomTask")

    # Test configuration - using RandomTask for multitask curriculum
    env_config = InternBootcampEnvConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=2,  # Small group for testing
        use_wandb=False,
        wandb_name="intern_bootcamp_random_test",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=2,
        steps_per_eval=0,
        max_token_length=2048,  # Increased for diverse tasks
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        # InternBootcamp specific settings - using RandomTask
        task_name="RandomTask",
        task_params={},  # RandomTask doesn't need specific params
        correct_reward=1.0,
        incorrect_reward=-0.5,
        format_bonus=0.2,
        require_reasoning=True,
        min_reasoning_length=20,
        temperature=0.7,
        top_p=0.9,
    )

    server_configs = [
        APIServerConfig(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        )
    ]

    logger.info("Using RandomTask configuration for multitask curriculum")
    logger.debug(f"Env Config: {env_config}")
    logger.debug(f"Server Configs: {server_configs}")

    try:
        env = InternBootcampEnv(
            config=env_config, server_configs=server_configs, slurm=False
        )
    except Exception as e:
        logger.exception(f"Failed to initialize InternBootcampEnv: {e}")
        return

    logger.info("Running RandomTask tests")
    try:
        await env.setup()

        # Test 1: Generate multiple random problems to show variety
        logger.info("\n========== Test 1: Multiple Random Problems ==========")

        for i in range(5):
            logger.info(f"\n--- Random Problem {i+1} ---")
            item = await env.get_next_item()
            prompt_tuple, metadata = item

            # Extract bootcamp name from identity if available
            bootcamp_name = "Unknown"
            if (
                isinstance(metadata["identity"], dict)
                and "_bootcamp_name" in metadata["identity"]
            ):
                bootcamp_name = metadata["identity"]["_bootcamp_name"]

            logger.info(f"  Selected Bootcamp: {bootcamp_name}")
            logger.info(f"  Task: {metadata['task_name']}")
            logger.info(f"  Prompt preview: {metadata['raw_prompt'][:150]}...")

        # Test 2: Collect and score trajectories from a random problem
        logger.info("\n========== Test 2: Trajectory Collection & Scoring ==========")
        item = await env.get_next_item()
        prompt_tuple, metadata = item

        # Extract bootcamp name
        bootcamp_name = "Unknown"
        if (
            isinstance(metadata["identity"], dict)
            and "_bootcamp_name" in metadata["identity"]
        ):
            bootcamp_name = metadata["identity"]["_bootcamp_name"]

        logger.info(f"Testing with bootcamp: {bootcamp_name}")
        logger.info(f"Problem: {metadata['raw_prompt'][:200]}...")

        # Collect trajectories
        scored_data, backlog = await env.collect_trajectories(item)
        logger.info(f"Collected and scored {len(scored_data['scores'])} responses")

        for i, score in enumerate(scored_data["scores"]):
            response_preview = (
                scored_data["messages"][i][-1]["content"][:100]
                if scored_data["messages"][i]
                else "No response"
            )
            logger.info(
                f"  Response {i+1}: Score={score:.2f}, Preview: {response_preview}..."
            )

        # Test 3: Quick evaluation with random tasks
        logger.info("\n========== Test 3: Random Task Evaluation ==========")

        async def quick_evaluate(*args, **kwargs):
            logger.info("Starting evaluation with random tasks")
            eval_tasks = []
            bootcamp_names = []

            for i in range(3):  # Only 3 problems for testing
                logger.info(f"Starting evaluation problem {i+1}/3")

                # Generate a problem to see which bootcamp is selected
                test_item = await env.get_next_item()
                _, test_metadata = test_item
                if (
                    isinstance(test_metadata["identity"], dict)
                    and "_bootcamp_name" in test_metadata["identity"]
                ):
                    bootcamp_name = test_metadata["identity"]["_bootcamp_name"]
                    bootcamp_names.append(bootcamp_name)
                    logger.info(f"  Evaluation problem {i+1} using: {bootcamp_name}")

                eval_tasks.append(env.evaluate_single_problem())

            results = await asyncio.gather(*eval_tasks)

            # Calculate metrics
            correct_count = sum(1 for is_correct, _ in results if is_correct)
            format_count = sum(1 for _, has_format in results if has_format)
            total_count = len(results)

            accuracy = correct_count / total_count if total_count > 0 else 0
            format_rate = format_count / total_count if total_count > 0 else 0

            logger.info("Evaluation complete:")
            logger.info(f"  Bootcamps used: {bootcamp_names}")
            logger.info(f"  Accuracy: {accuracy:.2%}")
            logger.info(f"  Format rate: {format_rate:.2%}")

            return [("eval/random_tasks_accuracy", accuracy)]

        env.evaluate = quick_evaluate
        await env.evaluate()

        # Test 4: Test specific bootcamp fallback
        logger.info("\n========== Test 4: Specific Bootcamp Test ==========")

        # Test with a specific bootcamp to ensure single-task mode still works
        specific_config = InternBootcampEnvConfig(
            **env_config.model_dump(),
            task_name="Game24bootcamp",
            task_params={
                "num_numbers": 4,
                "range_max": 20,
                "target_max": 30,
            },
        )

        try:
            specific_env = InternBootcampEnv(
                config=specific_config,
                server_configs=server_configs,
                slurm=False,
                testing=True,
            )

            await specific_env.setup()
            item = await specific_env.get_next_item()
            _, metadata = item

            logger.info("Specific bootcamp test (Game24bootcamp):")
            logger.info(f"  Task: {metadata['task_name']}")
            logger.info(f"  Problem: {metadata['identity']}")
            logger.info(f"  Prompt preview: {metadata['raw_prompt'][:100]}...")

        except Exception as e:
            logger.error(f"Failed to test specific bootcamp: {e}")

        # Test 5: Show bootcamp registry info
        logger.info("\n========== Test 5: Bootcamp Registry Info ==========")
        from environments.intern_bootcamp.bootcamp_registry import (
            get_available_bootcamps,
        )

        available = get_available_bootcamps()
        logger.info(f"Total available bootcamps: {len(available)}")
        logger.info(f"Sample bootcamps: {available[:10]}")

        # Show some variety in bootcamp names
        math_bootcamps = [
            name
            for name in available
            if any(x in name.lower() for x in ["math", "game", "number"])
        ]
        logic_bootcamps = [
            name
            for name in available
            if any(x in name.lower() for x in ["logic", "puzzle", "cipher"])
        ]

        logger.info(f"Math-related bootcamps (sample): {math_bootcamps[:5]}")
        logger.info(f"Logic-related bootcamps (sample): {logic_bootcamps[:5]}")

        logger.info("\n========== All Tests Complete ==========")
        logger.info("RandomTask multitask curriculum is working correctly!")

    except Exception as e:
        logger.exception(f"An error occurred during testing: {e}")


if __name__ == "__main__":
    asyncio.run(main())
