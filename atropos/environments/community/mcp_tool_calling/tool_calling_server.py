import json
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import wandb
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


class SingleToolCallingEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    @classmethod
    def config_init(self) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
            group_size=32,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="toolcall_think",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
            APIServerConfig(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                base_url="http://localhost:9005/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    async def create_rollout_table(self, wandb_metrics):

        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["text", "score", "expected_tool_call"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    table.add_data(item[0], item[1], item[2])
            wandb_metrics["train/rollouts"] = table

        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Log to wandb with comprehensive metrics.
        """
        if wandb_metrics is None:
            wandb_metrics = dict()

        # Try to calculate percent_correct, skip if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # Load the full dataset
        full_dataset = load_dataset(
            "NousResearch/XLAM-Atropos",
            "default",
            split="train",
        )

        full_dataset = full_dataset.shuffle(seed=42)

        # Create train/test split on the fly (e.g., 95% train, 5% test)
        split_dataset = full_dataset.train_test_split(test_size=0.02, seed=42)

        # Keep the splits as is - no need to reformat
        self.train = split_dataset["train"]
        self.test = split_dataset["test"]

        self.iter = 0

    async def rollout_and_score_eval(self, test_item):
        # Extract conversations from test item
        conversations = test_item["conversations"]

        # Find system message and human message
        system_message = next(
            (msg for msg in conversations if msg["from"] == "system"), None
        )
        human_message = next(
            (msg for msg in conversations if msg["from"] == "human"), None
        )
        expected_gpt_message = next(
            (msg for msg in conversations if msg["from"] == "gpt"), None
        )

        if not human_message or not expected_gpt_message:
            return 0  # Skip invalid conversations

        # Create messages for model
        messages = []
        if system_message:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt + "\n\n" + system_message["value"],
                }
            )
        messages.append({"role": "user", "content": human_message["value"]})

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get model completion using completion() instead of chat_completion()
        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=1024 * 15,
            temperature=1.0,
            split="eval",
        )

        # Extract the model's response from the completion
        model_response = completion.choices[0].text
        expected_response = expected_gpt_message["value"]

        # Extract and compare tool calls
        score = self._compare_tool_calls(model_response, expected_response)
        return score

    def _extract_tool_call_jsons(self, text):
        """
        Extract multiple JSONs from within <tool_call> tags

        Args:
            text: Text containing tool calls

        Returns:
            List of parsed JSON objects or empty list if extraction/parsing fails
        """
        # Find all content between <tool_call> tags
        matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        tool_calls = []

        for match in matches:
            try:
                # Parse the JSON content
                json_str = match
                tool_call = json.loads(json_str)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # Skip invalid JSON but continue processing other matches
                continue

        return tool_calls

    def _compare_tool_calls(self, model_response, expected_response):
        """
        Compare multiple tool calls by extracting JSONs from <tool_call> tags and comparing content

        Returns:
            1 if all tool calls match (all required calls are present with correct values), 0 otherwise
        """
        # Extract JSONs from tool calls
        model_jsons = self._extract_tool_call_jsons(model_response)
        expected_jsons = self._extract_tool_call_jsons(expected_response)

        # If we couldn't extract any JSONs or the count doesn't match, return 0
        if not model_jsons or not expected_jsons:
            return 0

        # Copy the expected_jsons to avoid modifying the original
        remaining_expected_jsons = expected_jsons.copy()

        # For each model JSON, try to find a matching expected JSON
        for model_json in model_jsons:
            found_match = False

            for i, expected_json in enumerate(remaining_expected_jsons):
                if self._json_objects_match(model_json, expected_json):
                    # Remove the matched expected JSON
                    remaining_expected_jsons.pop(i)
                    found_match = True
                    break

            # If no match was found for this model JSON, return 0
            if not found_match:
                return 0

        # If we've matched all expected JSONs (none remaining), return 1
        return 1 if not remaining_expected_jsons else 0

    def _json_objects_match(self, json1, json2):
        """
        Check if two JSON objects match, with all fields in json2 existing in json1
        with the same values.

        Args:
            json1: First JSON object
            json2: Second JSON object (expected values)

        Returns:
            True if objects match, False otherwise
        """
        try:
            # Check if all expected fields are in model response
            for key in json2:
                if key not in json1:
                    return False

                # For nested dictionaries (like 'arguments'), check all values
                if isinstance(json2[key], dict) and isinstance(json1[key], dict):
                    for arg_key in json2[key]:
                        if arg_key not in json1[key]:
                            return False
                        if json2[key][arg_key] != json1[key][arg_key]:
                            return False
                # For non-dictionary fields, check direct equality
                elif json2[key] != json1[key]:
                    return False

            # All checks passed
            return True
        except Exception:
            # Any error in comparison counts as failure
            return False

    async def evaluate(self, *args, **kwargs):
        eval_tasks = []
        for test_item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(test_item))
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        # Extract messages from the item
        messages = []
        for role_dict in item[0]:
            messages.append(dict(role_dict))

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get completions from the model using completion() instead of chat_completion()
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=1024 * 15,
            temperature=0.8,  # Using temperature to get diverse responses
        )

        to_score = list()

        for i, completion_choice in enumerate(completions.choices):
            # Create a copy of the prompt messages
            trajectory_messages = []
            for role_dict in item[0]:
                trajectory_messages.append(dict(role_dict))

            # Add the model's response
            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.text}
            )

            # Add to scoring queue with expected answer
            to_score.append(
                (
                    tuple(trajectory_messages),
                    item[1],  # The expected tool call JSON
                )
            )

        # Call score to get the scored data
        scored_data = await self.score(to_score)
        to_backlog = []

        return scored_data, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # Extract the expected JSONs from the answer
        expected_jsons = self._extract_tool_call_jsons(rollout_group_data[0][1])

        # If we can't extract the expected tool call JSONs, skip this item
        if not expected_jsons:
            return None

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            # Extract the model's response
            model_response = item[0][-1]["content"]

            # Score 1 if tool calls match, 0 otherwise
            reward = 1 if self._compare_tool_calls(model_response, item[1]) else 0

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)

            # Break once we have enough examples
            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Record success rate metrics
        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        # Apply length penalty if all responses are correct
        if all([score == 1.0 for score in scores["scores"]]):
            # Calculate token lengths
            token_lengths = [len(token) for token in scores["tokens"]]
            if max(token_lengths) == 0:
                # Edge case protection
                return None

            # Get max allowed token length from config
            max_allowed_length = self.config.max_token_length
            # Set threshold at 50% of max_token_length - no penalty below this
            length_threshold = max_allowed_length * 0.5

            # Apply modified length penalty with threshold
            scores["scores"] = []
            for length in token_lengths:
                if length <= length_threshold:
                    # No penalty for responses under threshold
                    scores["scores"].append(1.0)
                else:
                    # Calculate how far we are between threshold and max as a percentage
                    percentage_of_range = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    # Cap at 1.0 in case length exceeds max_allowed_length
                    percentage_of_range = min(percentage_of_range, 1.0)
                    # Apply linear penalty scaling from 1.0 down to 0.0
                    scores["scores"].append(1.0 - percentage_of_range)

        # Check if all scores are the same (no learning signal)
        if all(scores["scores"][0] == score for score in scores["scores"]):
            return None

        return scores

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1

        # Extract conversation elements
        conversations = next_item["conversations"]

        # Find system, human and gpt messages
        system_message = next(
            (msg for msg in conversations if msg["from"] == "system"), None
        )
        human_message = next(
            (msg for msg in conversations if msg["from"] == "human"), None
        )
        expected_gpt_message = next(
            (msg for msg in conversations if msg["from"] == "gpt"), None
        )

        # Create prompt tuple using frozensets as required
        prompt = []
        if system_message:
            # Combine our base system prompt with the dataset-specific system message
            combined_system_content = system_prompt + "\n\n" + system_message["value"]
            prompt.append(
                frozenset(
                    {"role": "system", "content": combined_system_content}.items()
                )
            )

        # Add user message
        if human_message:
            prompt.append(
                frozenset({"role": "user", "content": human_message["value"]}.items())
            )

        # Return expected assistant response (the tool call JSON) as the "answer"
        answer = expected_gpt_message["value"] if expected_gpt_message else ""

        return (tuple(prompt), answer)

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):

        # save rollout to trajectory
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored_data["tokens"][i]),
                    scored_data["scores"][i],
                    item[1],  # Just keep the expected tool call JSON
                )
                for i in range(num_keep)
            ]
        )
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)


if __name__ == "__main__":
    SingleToolCallingEnv.cli()
