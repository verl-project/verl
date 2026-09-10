#!/usr/bin/env python3
"""
Run Wikipedia Article Creator with OpenAI models

This script demonstrates how to use the WikipediaArticleCreatorEnv with OpenAI models
for research and article generation.

Usage:
    python run_with_openai.py [--topic "Your Research Topic"] [--model "gpt-4o"] [--max-steps 10]

Requirements:
    - A .env file with OPENAI_API_KEY and TAVILY_API_KEY (see .env.template)
    - All dependencies installed (openai, tavily-python, python-dotenv)
"""

import argparse
import asyncio
import logging
import os
import sys

# Add the parent directory to the path so we can import the environment
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum  # noqa: E402
from environments.community.wikipedia_research.wikipedia_article_creator import (  # noqa: E402
    WikipediaArticleCreatorConfig,
    WikipediaArticleCreatorEnv,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wikipedia Article Creator with OpenAI"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="Climate change in Antarctica",
        help="Research topic for article creation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_NAME", "gpt-4o"),
        help="OpenAI model to use (default: gpt-4o or MODEL_NAME from .env)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("MAX_STEPS", "10")),
        help="Maximum research steps (default: 10 or MAX_STEPS from .env)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", "0.7")),
        help="Temperature setting (default: 0.7 or TEMPERATURE from .env)",
    )
    parser.add_argument(
        "--debug-output",
        type=str,
        default="",
        help="Path to save raw model responses for debugging (optional)",
    )
    return parser.parse_args()


async def main():
    """Run the environment with specified model and topic"""
    args = parse_args()

    # Validate environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not openai_api_key:
        logger.error("OPENAI_API_KEY is missing. Please add it to your .env file.")
        return

    if not tavily_api_key:
        logger.error("TAVILY_API_KEY is missing. Please add it to your .env file.")
        return

    # Create custom configuration
    env_config = WikipediaArticleCreatorConfig(
        tokenizer_name="gpt2",  # Use a standard HuggingFace tokenizer for token counting
        group_size=1,
        use_wandb=False,  # Set to True if you want to use wandb
        rollout_server_url="",  # Empty string for direct API access
        total_steps=1,
        batch_size=1,
        steps_per_eval=1,
        max_token_length=8192,
        inference_weight=1.0,
        wandb_name="wikipedia_article_creator_test",
        eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
        eval_limit_ratio=0.1,
        max_steps=args.max_steps,
        temperature=args.temperature,
        thinking_active=True,
        eval_topics=1,
        tool_timeout=30.0,
        tavily_api_key=tavily_api_key,
        min_article_sections=3,
        max_article_tokens=4096,
        topics_file="topics.json",
        logging_active=True,
    )

    # Configure OpenAI server
    server_configs = [
        APIServerConfig(
            model_name=args.model,
            base_url="https://api.openai.com/v1",  # Explicitly set OpenAI base URL
            api_key=openai_api_key,
            num_max_requests_at_once=1,
            num_requests_for_eval=1,
        ),
    ]

    # Initialize the environment with our configs
    logger.info(f"Initializing environment with model: {args.model}")
    env = WikipediaArticleCreatorEnv(
        env_config, server_configs, slurm=False, testing=True
    )
    await env.setup()

    # Run a single episode with the specified topic
    topic = args.topic
    episode_id = 1

    logger.info(f"Starting research on topic: {topic}")
    logger.info(f"Maximum steps: {args.max_steps}, Temperature: {args.temperature}")

    episode = env._get_or_create_episode(episode_id, topic)

    # Run until terminal state
    step_count = 0
    while not episode.is_terminal:
        step_count += 1
        logger.info(f"Executing step {step_count}...")

        is_terminal, step_data = await env._next_step(episode)
        tool_calls = step_data.get("tool_calls", [])

        logger.info(f"Step {episode.steps_taken} completed")
        logger.info(f"Tool calls made: {len(tool_calls)}")

        # Extract tool names for logging
        if tool_calls:
            tool_names = [tool.get("name", "unknown") for tool in tool_calls]
            logger.info(f"Tools used: {', '.join(tool_names)}")

        if is_terminal:
            if episode.final_article:
                logger.info("Article generated successfully!")
                logger.info(f"Article length: {len(episode.final_article)} characters")

                # Create a sanitized filename
                safe_topic = "".join(
                    c if c.isalnum() or c in " _-" else "_" for c in topic
                )
                filename = f"article_{safe_topic.replace(' ', '_')}.md"

                # Save the article to a file
                with open(filename, "w") as f:
                    f.write(episode.final_article)
                logger.info(f"Article saved to {filename}")

                # Evaluate article quality
                quality_metrics = env._assess_article_quality(
                    episode.final_article, episode.research_facts
                )
                logger.info(f"Article quality metrics: {quality_metrics}")
            else:
                logger.warning("Episode terminated without producing an article")

    logger.info(f"Research complete. Steps taken: {episode.steps_taken}")
    logger.info(f"Research facts collected: {len(episode.research_facts)}")


if __name__ == "__main__":
    asyncio.run(main())
