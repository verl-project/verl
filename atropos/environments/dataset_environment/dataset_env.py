import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import load_dataset
from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.envs.reward_fns import registry
from atroposlib.envs.reward_fns.combined_reward import CombinedReward
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetEnvConfig(BaseEnvConfig):
    dataset_name: str = Field(..., description="HuggingFace dataset name")
    dataset_config: Optional[str] = Field(
        None, description="Dataset configuration name"
    )
    split: str = Field("train", description="Dataset split to use")
    dataset_path: Optional[str] = Field(
        None, description="Local path to dataset (alternative to dataset_name)"
    )
    prompt_field: str = Field(..., description="Field in dataset to use as prompt")
    answer_field: Optional[str] = Field(
        None, description="Field in dataset to use as answer"
    )
    ground_truth_field: Optional[str] = Field(
        None, description="Field in dataset containing canonical correct answer"
    )
    system_prompt: Optional[str] = Field(None, description="System prompt to use")
    prefill: Optional[str] = Field(
        None, description="Text to prefill the completion with (e.g. '<think>')"
    )
    shuffle_dataset: bool = Field(True, description="Whether to shuffle the dataset")
    max_generations_per_prompt: int = Field(
        1, description="Number of generations per prompt for collection"
    )
    include_messages_in_scoring: bool = Field(
        False, description="Whether to include messages in scoring"
    )
    reward_funcs: List[str] = Field(
        default_factory=list,
        description="List of reward function names to apply (legacy)",
    )
    reward_functions: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of reward functions to apply (string names or full configs)",
    )

    # Completion parameters
    temperature: float = Field(0.7, description="Temperature for generation")
    top_p: float = Field(0.9, description="Top-p for generation")
    max_tokens: int = Field(4096, description="Maximum tokens for generation")
    length_warmup_steps: int = Field(0, description="Steps for length warmup")
    min_tokens: int = Field(0, description="Minimum tokens for generation")

    eval_dataset_name: Optional[str] = Field(
        None, description="Evaluation dataset name"
    )
    eval_dataset_config: Optional[str] = Field(
        None, description="Evaluation dataset config"
    )
    eval_split: Optional[str] = Field(None, description="Evaluation dataset split")


class DatasetEnv(BaseEnv):
    def __init__(
        self, config: DatasetEnvConfig, server_configs, slurm=True, testing=False
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config = config
        self.dataset = None
        self.iter = 0
        self.metric_buffer = {}

        self.reward_function = self._initialize_reward_function()

    def _initialize_reward_function(self):
        if hasattr(self.config, "reward_functions") and self.config.reward_functions:
            if len(self.config.reward_functions) == 1:
                return registry.create(self.config.reward_functions[0])
            else:
                return CombinedReward(
                    rewards=self.config.reward_functions, normalization="sum"
                )
        elif hasattr(self.config, "reward_funcs") and self.config.reward_funcs:
            if len(self.config.reward_funcs) == 1:
                return registry.create(self.config.reward_funcs[0])
            else:
                return CombinedReward(
                    rewards=self.config.reward_funcs, normalization="none"
                )

    async def setup(self):
        if self.config.dataset_path:
            self.dataset = load_dataset(
                self.config.dataset_path, split=self.config.split
            )
        else:
            self.dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=self.config.split,
            )
        logger.info(f"Dataset features: {self.dataset.features}")
        logger.info(f"Sample item keys: {list(self.dataset[0].keys())}")
        logger.info(f"Sample item: {self.dataset[0]}")

        if self.config.shuffle_dataset:
            self.dataset = self.dataset.shuffle()

        self.metric_buffer = {}

    async def get_next_item(self) -> Item:
        if not self.dataset:
            await self.setup()

        item = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1

        user_msg = {"role": "user", "content": item[self.config.prompt_field]}
        prompt = tuple([frozenset(user_msg.items())])

        answer = None
        if self.config.answer_field and self.config.answer_field in item:
            answer = item[self.config.answer_field]

        ground_truth = None
        if self.config.ground_truth_field and self.config.ground_truth_field in item:
            ground_truth = item[self.config.ground_truth_field]

        return (prompt, answer, ground_truth)

    async def collect_trajectory(self, item: Item) -> Tuple[List, List]:
        # Extract user prompt and answer from item
        user_content = dict(item[0][0])["content"]
        answer = item[1] if len(item) > 1 else None

        # Create messages list
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        messages.append({"role": "user", "content": user_content})

        # Add prefill as assistant message if configured
        if self.config.prefill:
            messages.append({"role": "assistant", "content": self.config.prefill})

        # Convert messages to a prompt string using the tokenizer
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Calculate max tokens for generation (with optional warmup)
        max_tokens = self.config.max_tokens
        if self.config.length_warmup_steps > 0:
            warmup_progress = min(1.0, self.curr_step / self.config.length_warmup_steps)
            max_tokens = int(
                self.config.min_tokens
                + warmup_progress * (self.config.max_tokens - self.config.min_tokens)
            )

        # Generate completion using completions API
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.max_generations_per_prompt,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        to_score = []
        to_backlog = []

        # Process completions
        for completion in completions.choices:
            # Get the completion text
            completion_text = (
                completion.text
                if hasattr(completion, "text")
                else completion.message.content
            )

            # Build full message sequence for scoring
            full_messages = []
            if self.config.system_prompt:
                full_messages.append(
                    {"role": "system", "content": self.config.system_prompt}
                )

            full_messages.append({"role": "user", "content": user_content})

            # Combine prefill with completion if prefill was used
            response_content = completion_text
            if self.config.prefill:
                response_content = self.config.prefill + completion_text

            full_messages.append({"role": "assistant", "content": response_content})

            # Add to scoring list with answer and ground truth
            to_score.append((full_messages, answer, item[2] if len(item) > 2 else None))

        return to_score, to_backlog

    async def postprocess_histories(self, trajectories: List) -> Tuple[List, List]:
        return trajectories, []

    async def collect_trajectories(self, item: Item) -> Tuple[List, List]:
        self.current_item = item

        # Extract user prompt from item
        user_content = dict(item[0][0])["content"]

        # Create messages list
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        messages.append({"role": "user", "content": user_content})

        # Add prefill as assistant message if configured
        if self.config.prefill:
            messages.append({"role": "assistant", "content": self.config.prefill})

        # Convert messages to a prompt string using the tokenizer
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Calculate max tokens for generation (with optional warmup)
        max_tokens = self.config.max_tokens

        # Generate completions
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        print(f"Completions: {completions}")
        # Process completions
        trajectories = []
        for completion in completions.choices:
            # Get the completion text
            completion_text = (
                completion.text
                if hasattr(completion, "text")
                else completion.message.content
            )

            # Build complete message sequence
            full_messages = []
            if self.config.system_prompt:
                full_messages.append(
                    {"role": "system", "content": self.config.system_prompt}
                )

            full_messages.append({"role": "user", "content": user_content})

            # Combine prefill with completion if prefill was used
            response_content = completion_text
            if self.config.prefill:
                response_content = self.config.prefill + completion_text

            full_messages.append({"role": "assistant", "content": response_content})

            trajectories.append(full_messages)

        return trajectories, []

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        logger.warning(f"Scoring {len(rollout_group_data)} rollout items")

        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["advantages"] = None
        scores["ref_logprobs"] = None
        scores["messages"] = None if not self.config.include_messages_in_scoring else []

        answer = (
            self.current_item[1]
            if self.current_item and len(self.current_item) > 1
            else None
        )
        logger.warning(f"Answer for current item: {answer}")

        ground_truth = (
            self.current_item[2]
            if self.current_item and len(self.current_item) > 2
            else None
        )
        logger.warning(f"Ground truth for current item: {ground_truth}")

        formatted_completions = []
        for trajectory in rollout_group_data:
            if trajectory and isinstance(trajectory, list):
                assistant_messages = [
                    msg
                    for msg in trajectory
                    if isinstance(msg, dict) and msg.get("role") == "assistant"
                ]
                if assistant_messages:
                    formatted_completions.append([assistant_messages[-1]])

        if not formatted_completions:
            logger.warning("No valid completions to score")
            return None

        try:
            reward_kwargs = {
                "solution": answer,
                "ground_truth": ground_truth,
                "item": self.current_item,
                "config": self.config,
            }

            all_rewards = self.reward_function(formatted_completions, **reward_kwargs)

            logger.info(f"Calculated rewards: {all_rewards}")

        except Exception as e:
            logger.error(f"Error applying reward functions: {e}")
            logger.exception(e)
            all_rewards = [0.0] * len(formatted_completions)

        for i, (trajectory, reward) in enumerate(zip(rollout_group_data, all_rewards)):
            try:
                tokenized = tokenize_for_trainer(self.tokenizer, trajectory)

                scores["tokens"].append(tokenized["tokens"])
                scores["masks"].append(tokenized["masks"])
                scores["scores"].append(reward)

                if self.config.include_messages_in_scoring:
                    if "messages" not in scores:
                        scores["messages"] = []
                    scores["messages"].append(trajectory)
                logger.warning(f"Scores: {scores['scores']}")
            except Exception as e:
                logger.error(f"Error processing trajectory {i}: {e}")
                logger.exception(e)

        if not scores["tokens"]:
            logger.warning("No valid scores generated")
            return None

        logger.info(f"Generated scores: {scores['scores']}")
        return scores

    async def evaluate(self):
        if (
            not hasattr(self.config, "eval_dataset_name")
            or not self.config.eval_dataset_name
        ):
            return

        if not hasattr(self, "eval_dataset"):
            self.eval_dataset = load_dataset(
                self.config.eval_dataset_name,
                self.config.eval_dataset_config,
                split=self.config.eval_split,
            )
            self.eval_dataset = self.eval_dataset.select(
                range(min(100, len(self.eval_dataset)))
            )

        eval_metrics = {}
        eval_tasks = []

        for i in range(min(self.config.max_eval_workers, len(self.eval_dataset))):
            item = self.eval_dataset[i]
            user_msg = {"role": "user", "content": item[self.config.prompt_field]}
            prompt = tuple([frozenset(user_msg.items())])

            answer = None
            if self.config.answer_field and self.config.answer_field in item:
                answer = item[self.config.answer_field]

            eval_tasks.append(self.collect_trajectory((prompt, answer)))

        eval_results = await asyncio.gather(*eval_tasks)

        eval_scores = []
        for result in eval_results:
            if result[0]:
                scored_data = await self.score(result[0])
                if scored_data and "scores" in scored_data:
                    eval_scores.extend(scored_data["scores"])

        if eval_scores:
            eval_metrics["eval/mean_score"] = sum(eval_scores) / len(eval_scores)
            eval_metrics["eval/max_score"] = max(eval_scores)
            eval_metrics["eval/min_score"] = min(eval_scores)

        await self.wandb_log(eval_metrics)

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        metrics = wandb_metrics or {}

        for key, values in self.metric_buffer.items():
            if values:
                metrics[f"train/{key}"] = sum(values) / len(values)

        self.metric_buffer = {k: [] for k in self.metric_buffer}

        if hasattr(self, "reward_function") and self.wandb:
            if hasattr(self.reward_function, "set_wandb_logger"):
                self.reward_function.set_wandb_logger(self.wandb)

        await super().wandb_log(metrics)


if __name__ == "__main__":
    # Launch the DatasetEnv via the BaseEnv CLI (serve or process)
    DatasetEnv.cli()
