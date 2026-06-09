#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig
from atroposlib.envs.reward_fns import registry

# from atroposlib.utils.config_handler import ConfigHandler
from environments.dataset_environment.dataset_env import DatasetEnv, DatasetEnvConfig

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dataset environment local server")
    parser.add_argument(
        "--config",
        type=str,
        default="dataset_local",
        help="Configuration file name (without .yaml extension) relative to environments/dataset_environment/configs/,"
        " or full path to a YAML file.",
    )
    return parser.parse_args()


async def main():
    logger.info("Starting Dataset environment local server")

    # Parse command line arguments
    args = parse_arguments()

    # Initialize config handler
    # config_handler = ConfigHandler()

    # Determine config path
    if (
        os.path.isabs(args.config)
        or "/" in args.config
        or args.config.endswith(".yaml")
    ):
        config_path = args.config
    else:
        # Assume it's a name relative to the new default directory
        config_path = os.path.join(
            os.path.dirname(__file__), "configs", f"{args.config}.yaml"
        )

    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, "r") as f:
            import yaml

            raw_config = yaml.safe_load(f)
            logger.info("Loaded configuration successfully")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        logger.info("Ensure the --config argument is correct or the file exists.")
        return
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return

    # Ensure dataset configuration exists (assuming it's top-level in these files)
    if "dataset" not in raw_config:
        logger.warning(
            "'dataset' key not found at the top level of the config file. "
            "Assuming the entire file is the dataset configuration."
        )
        # Treat the whole raw_config as the 'dataset' section for compatibility
        dataset_section = raw_config
    else:
        dataset_section = raw_config["dataset"]

    if "dataset_name" not in dataset_section:
        logger.error("dataset_name not found in dataset configuration")
        return
    if "prompt_field" not in dataset_section:
        logger.error("prompt_field not found in dataset configuration")
        return

    # Configure the dataset environment
    # Merging logic: Start with raw_config defaults, then dataset_section specifics
    env_config_data = {**raw_config, **dataset_section}

    # Pydantic will ignore extra fields, so we just pass everything
    try:
        env_config = DatasetEnvConfig(**env_config_data)
    except Exception as pydantic_error:
        logger.error(f"Error validating configuration: {pydantic_error}")
        return

    # Preload reward functions
    reward_names_to_load = set()
    if env_config.reward_funcs:
        reward_names_to_load.update(env_config.reward_funcs)
    if env_config.reward_functions:
        for rf_config in env_config.reward_functions:
            if isinstance(rf_config, str):
                reward_names_to_load.add(rf_config)
            elif isinstance(rf_config, dict) and "type" in rf_config:
                reward_names_to_load.add(rf_config["type"])

    if reward_names_to_load:
        logger.info(f"Preloading reward functions: {list(reward_names_to_load)}")
        for func_name in reward_names_to_load:
            try:
                registry.get(func_name)
                logger.info(f"Successfully loaded reward function: {func_name}")
            except Exception as e:
                logger.error(f"Failed to load reward function {func_name}: {e}")

    # Server configuration - process env vars
    server_configs = []

    if "server_configs" in raw_config:
        for server_config in raw_config["server_configs"]:
            api_key = server_config.get("api_key", os.environ.get("OPENAI_API_KEY"))
            # Handle environment variable references like ${OPENAI_API_KEY}
            if (
                isinstance(api_key, str)
                and api_key.startswith("${")
                and api_key.endswith("}")
            ):
                env_var = api_key[2:-1]
                api_key = os.environ.get(env_var, "")

            server_configs.append(
                APIServerConfig(
                    model_name=server_config.get("model_name", "gpt-4.1-nano"),
                    base_url=server_config.get("base_url", None),
                    api_key=api_key,
                    timeout=server_config.get("timeout", 600),
                )
            )
    else:
        # Default configuration if not specified in config file
        logger.warning(
            "No 'server_configs' found in config. Using default OpenAI config."
        )
        server_configs.append(
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url=None,
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=600,
            )
        )

    # Create the environment
    logger.info("Creating dataset environment...")
    env = DatasetEnv(
        config=env_config,
        server_configs=server_configs,
        slurm=False,
    )

    # Setup the environment directly
    try:
        await env.setup()
        logger.info("Environment setup complete")
    except Exception as setup_error:
        logger.error(f"Error during environment setup: {setup_error}")
        return

    # --- Start Test Run --- #
    logger.info("\n=== Starting Local Test Run ===")
    test_items_count = 5
    successful_runs = 0

    for i in range(test_items_count):
        logger.info(f"\n--- Running Test Item {i+1}/{test_items_count} ---")
        try:
            # Get a sample item from the dataset
            item = await env.get_next_item()
            if not item or not item[0]:
                logger.warning("Failed to get a valid item from the environment.")
                continue

            prompt, answer, ground_truth = item
            user_content = dict(prompt[0])["content"]
            logger.info(
                f"Prompt: {user_content[:200]}..." if user_content else "(Empty Prompt)"
            )
            if answer:
                logger.info(
                    f"Answer: {answer[:200]}..." if answer else "(Empty Answer)"
                )
            if ground_truth:
                logger.info(
                    f"Ground Truth: {ground_truth[:200]}..."
                    if ground_truth
                    else "(Empty Ground Truth)"
                )

            # Collect trajectories (using group_size from config)
            logger.info(
                f"Collecting {env_config.group_size} trajectories for this item..."
            )
            trajectories_data, backlog = await env.collect_trajectories(item)

            if not trajectories_data:
                logger.warning("No trajectories were collected.")
                continue

            logger.info(f"Collected {len(trajectories_data)} trajectories.")
            # Log first trajectory message content for inspection
            if trajectories_data[0] and isinstance(trajectories_data[0], list):
                first_response = "(Empty or invalid trajectory format)"
                assistant_msgs = [
                    m
                    for m in trajectories_data[0]
                    if isinstance(m, dict) and m.get("role") == "assistant"
                ]
                if assistant_msgs:
                    first_response = assistant_msgs[-1].get("content", "(No content)")
                logger.info(f"First Response Content: {first_response[:300]}...")

            # Score the collected trajectories
            logger.info("Scoring trajectories...")
            scored_data = await env.score(trajectories_data)

            # Print scores
            if scored_data and "scores" in scored_data:
                scores_list = scored_data["scores"]
                logger.info(f"Scores: {scores_list}")
                logger.info(f"  Avg Score: {sum(scores_list)/len(scores_list):.4f}")
                successful_runs += 1
            else:
                logger.warning("No scores available in the scored data for this item.")

        except Exception as run_error:
            logger.error(f"Error during test item {i+1}: {run_error}")
            # Optionally continue to the next item or break
            # break

    logger.info(
        f"\n=== Local Test Run Complete ({successful_runs}/{test_items_count} items processed successfully) ==="
    )


if __name__ == "__main__":
    asyncio.run(main())
