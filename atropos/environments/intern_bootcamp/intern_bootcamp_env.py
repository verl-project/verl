#!/usr/bin/env python3
"""
InternBootcamp RL Environment for Atropos

This environment integrates InternBootcamp's verifiable reasoning tasks with the Atropos
RL training framework. It supports training on single tasks, with plans for multi-task
and curriculum learning modes.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from .bootcamp_registry import create_bootcamp, get_available_bootcamps

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# System prompt for reasoning tasks
SYSTEM_PROMPT = (
    "You are a deep thinking AI with strong reasoning abilities. You may use "
    "extremely long chains of thought to deeply consider the problem and "
    "deliberate with yourself via systematic reasoning processes to help come "
    "to a correct solution.\n\n"
    "You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem. "
    "Please think in English, even if the problem is presented in another "
    "language.\n\n"
    "When solving problems:\n"
    "1. Think step by step through the problem inside <think> tags\n"
    "2. Show your work clearly in your thinking\n"
    "3. Verify your answer before finalizing\n"
    "4. Follow the specific answer format requested in the problem\n\n"
    "Pay close attention to how the problem asks you to format your answer - "
    "some may require specific tags, notations, or formats."
)


class InternBootcampEnvConfig(BaseEnvConfig):
    """Configuration for the InternBootcamp environment."""

    # Task selection
    task_name: str = "RandomTask"  # Random task selection mode

    # Task-specific parameters
    task_params: Dict[str, Any] = {}

    # Reward configuration
    correct_reward: float = 1.0
    incorrect_reward: float = -0.5
    format_bonus: float = 0.2

    # Training parameters
    require_reasoning: bool = True
    min_reasoning_length: int = 50
    temperature: float = 0.7
    top_p: float = 0.9


class InternBootcampEnv(BaseEnv):
    """Environment for training on InternBootcamp reasoning tasks."""

    name = "intern_bootcamp"

    def __init__(
        self,
        config: InternBootcampEnvConfig,
        server_configs: Union[List[APIServerConfig], APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config = config

        # Task tracking
        self.bootcamp_instance = None
        self.current_task_name = config.task_name

        # Performance tracking
        self.task_correct_buffer = []
        self.format_correct_buffer = []
        self.eval_metrics = []

        self.system_prompt = SYSTEM_PROMPT

    async def setup(self):
        """Initialize the environment and bootcamp task."""
        logger.info(f"Setting up InternBootcampEnv with task: {self.config.task_name}")

        # Log available bootcamps
        available = get_available_bootcamps()
        logger.info(f"Found {len(available)} available bootcamp tasks")
        logger.debug(f"Available tasks (first 20): {available[:20]}")

        # Initialize the bootcamp task
        self._initialize_bootcamp()

        # Generate some test problems to verify setup
        try:
            for i in range(3):
                identity = self.bootcamp_instance.case_generator()
                prompt = self.bootcamp_instance.prompt_func(identity)
                logger.info(f"Test problem {i+1}: {prompt[:100]}...")
        except Exception as e:
            logger.error(f"Failed to generate test problems: {e}")
            raise

    def _initialize_bootcamp(self):
        """Initialize the bootcamp instance based on task name."""
        try:
            # Create bootcamp instance using the registry
            self.bootcamp_instance = create_bootcamp(
                self.config.task_name, **self.config.task_params
            )
            logger.info(
                f"Initialized {self.config.task_name} with params: {self.config.task_params}"
            )
        except ValueError as e:
            # If task not found, list available tasks
            available = get_available_bootcamps()
            logger.error(f"Task '{self.config.task_name}' not found!")
            logger.error(f"Available tasks (showing first 20): {available[:20]}")
            raise e
        except Exception as e:
            logger.error(f"Failed to initialize bootcamp: {e}")
            raise

    async def get_next_item(self) -> Tuple[Any, Dict]:
        """Get the next problem from the bootcamp."""
        # Generate a new problem
        identity = self.bootcamp_instance.case_generator()
        prompt = self.bootcamp_instance.prompt_func(identity)

        # Log which bootcamp is being used if RandomTask
        if (
            self.config.task_name == "RandomTask"
            and isinstance(identity, dict)
            and "_bootcamp_name" in identity
        ):
            logger.info(f"RandomTask selected: {identity['_bootcamp_name']}")

        # Create the message format expected by Atropos
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Return item with metadata
        return (
            messages,
            {
                "identity": identity,
                "task_name": self.current_task_name,
                "raw_prompt": prompt,
            },
        )

    async def collect_trajectories(self, item) -> Tuple[List, List]:
        """Collect trajectories for the current item."""
        messages, metadata = item
        logger.info(f"Collecting trajectories for item: {messages}")

        # Get completions from the model using chat_completion
        completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        to_score = []

        for i, completion in enumerate(completions.choices):
            model_response = completion.message.content

            # Create full conversation for scoring
            full_messages = messages + [
                {"role": "assistant", "content": model_response}
            ]

            to_score.append((full_messages, metadata, model_response))

        # Score the trajectories immediately and return a ScoredDataGroup
        scored_data = await self.score(to_score)
        backlog = []  # No backlog items for now

        return scored_data, backlog

    async def score(self, rollout_group_data) -> ScoredDataGroup:
        """Score the collected trajectories using bootcamp verification."""
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["messages"] = []

        for messages, metadata, model_response in rollout_group_data:
            # Verify the response using the bootcamp
            identity = metadata["identity"]

            # Calculate base score from bootcamp verification
            base_score = self.bootcamp_instance.verify_score(
                model_response,
                identity,
                format_score=self.config.format_bonus,
                short_penalty=self.config.require_reasoning,
                short_threshold=self.config.min_reasoning_length,
            )

            # Apply reward scaling
            if base_score >= 1.0:
                # Correct answer with format
                final_score = self.config.correct_reward
                self.task_correct_buffer.append(1)
                self.format_correct_buffer.append(1)
            elif base_score > 0:
                # Correct format but wrong answer
                final_score = self.config.incorrect_reward + base_score
                self.task_correct_buffer.append(0)
                self.format_correct_buffer.append(1)
            else:
                # Wrong answer and/or format
                final_score = self.config.incorrect_reward
                self.task_correct_buffer.append(0)
                self.format_correct_buffer.append(0)

            # Log the scoring details
            logger.debug(
                f"Scored response: base_score={base_score}, "
                f"final_score={final_score}, "
                f"identity={identity}"
            )

            # Tokenize for trainer
            tokens_dict = tokenize_for_trainer(
                self.tokenizer,
                messages,
                None,
            )

            scored_data["tokens"].append(tokens_dict["tokens"])
            scored_data["masks"].append(tokens_dict["masks"])
            scored_data["scores"].append(final_score)
            scored_data["messages"].append(messages)

        return scored_data

    async def evaluate(self, *args, **kwargs):
        """Evaluate the model on test problems."""
        logger.info(f"Starting evaluation for {self.current_task_name}")

        eval_tasks = []
        num_eval_problems = 20  # Number of problems to evaluate on

        # Generate evaluation problems
        for i in range(num_eval_problems):
            eval_tasks.append(self.evaluate_single_problem())

        # Run evaluations in parallel
        results = await asyncio.gather(*eval_tasks)

        # Calculate metrics
        correct_count = sum(1 for is_correct, _ in results if is_correct)
        format_count = sum(1 for _, has_format in results if has_format)
        total_count = len(results)

        accuracy = correct_count / total_count if total_count > 0 else 0
        format_rate = format_count / total_count if total_count > 0 else 0

        logger.info(
            f"Evaluation complete: accuracy={accuracy:.2%}, "
            f"format_rate={format_rate:.2%} "
            f"({correct_count}/{total_count} correct)"
        )

        # Store metrics for wandb logging
        self.eval_metrics.append((f"eval/{self.current_task_name}_accuracy", accuracy))
        self.eval_metrics.append(
            (f"eval/{self.current_task_name}_format_rate", format_rate)
        )
        self.eval_metrics.append(("eval/overall_accuracy", accuracy))

        return self.eval_metrics

    async def evaluate_single_problem(self) -> Tuple[bool, bool]:
        """Evaluate a single problem."""
        try:
            # Generate a problem
            identity = self.bootcamp_instance.case_generator()
            prompt = self.bootcamp_instance.prompt_func(identity)

            # Create messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Get model response using chat_completion
            completion = await self.server.chat_completion(
                messages=messages,
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.0,  # Deterministic for evaluation
                top_p=1.0,
                split="eval",
            )

            model_response = completion.choices[0].message.content

            # Score the response
            score = self.bootcamp_instance.verify_score(
                model_response,
                identity,
                format_score=self.config.format_bonus,
                short_penalty=False,  # Don't penalize short responses in eval
            )

            is_correct = score >= 1.0
            has_format = score > 0

            return is_correct, has_format

        except Exception as e:
            logger.error(f"Error evaluating problem: {e}")
            return False, False

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add training metrics
        if self.task_correct_buffer:
            wandb_metrics[f"train/{self.current_task_name}_accuracy"] = sum(
                self.task_correct_buffer
            ) / len(self.task_correct_buffer)

        if self.format_correct_buffer:
            wandb_metrics[f"train/{self.current_task_name}_format_rate"] = sum(
                self.format_correct_buffer
            ) / len(self.format_correct_buffer)

        # Add evaluation metrics
        for metric_name, value in self.eval_metrics:
            wandb_metrics[metric_name] = value

        # Clear buffers
        self.task_correct_buffer = []
        self.format_correct_buffer = []
        self.eval_metrics = []

        await super().wandb_log(wandb_metrics)

    @classmethod
    def config_init(cls) -> Tuple[InternBootcampEnvConfig, List[APIServerConfig]]:
        """Initialize environment and server configurations."""
        env_config = InternBootcampEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            max_num_workers=64,
            rollout_server_url="http://localhost:8000",
            total_steps=10000,
            batch_size=1024,
            steps_per_eval=100,
            max_token_length=16384,
            inference_weight=1.0,
            wandb_name="intern_bootcamp_random_tasks",
            data_path_to_save_groups="data/intern_bootcamp_random_tasks.jsonl",
            # Task configuration
            task_name="RandomTask",
            task_params={},
            # Reward configuration
            correct_reward=1.0,
            incorrect_reward=-0.5,
            format_bonus=0.2,
            # Training parameters
            require_reasoning=True,
            min_reasoning_length=50,
            temperature=0.7,
            top_p=0.9,
        )

        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=64,
            )
        ]

        return env_config, server_configs


if __name__ == "__main__":
    InternBootcampEnv.cli()
