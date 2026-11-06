import json
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
    "You are an AI assistant capable of using tools to answer requests. "
    "When a tool is required, you must generate a single JSON object specifying the tool and its arguments. "
    'The JSON format is: {"tool_name": "<tool_name>", "arguments": {<key_value_args>}}. '
    "Do not output any text before or after this JSON object. "
    "You may use <think></think> tags for your internal reasoning before producing the JSON output."
)


class McpEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
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
            dataset_path="my_mcp_dataset.json",  # ADDED: Path to your JSON dataset
            num_rollouts_per_group_for_logging=4,  # Added for logging
            num_rollouts_to_keep=4,  # Added for logging
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
        wandb_metrics = await self.create_rollout_table(wandb_metrics)  # Moved here
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # Load the full dataset
        # full_dataset = load_dataset( # Old way, replaced

        #     "NousResearch/XLAM-Atropos",
        #     "default",
        #     streaming=False,
        #     split="train",
        # )
        with open(self.config.dataset_path, "r") as f:  # Load dataset from path
            _ = json.load(f)

        full_dataset = load_dataset(
            "json", data_files={"train": self.config.dataset_path}
        )[
            "train"
        ]  # Load JSON directly. This is for using 'datasets' methods
        # full_dataset = full_dataset.shuffle(seed=42) # shuffle here
        full_dataset = full_dataset.shuffle(seed=42)  # Shuffling datasets object

        # Create train/test split on the fly (e.g., 95% train, 5% test)
        split_dataset = full_dataset.train_test_split(test_size=0.02, seed=42)

        # Keep the splits as is - no need to reformat
        self.train = split_dataset["train"]
        self.test = split_dataset["test"]

        self.iter = 0

    async def rollout_and_score_eval(self, test_item_data):
        user_prompt_content = test_item_data["user_prompt_text"]
        expected_mcp_call_dict = test_item_data["expected_mcp_call"]

        messages = [{"role": "user", "content": user_prompt_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        completion = await self.server.completion(prompt=prompt, n=1, max_tokens=1024)
        model_response = completion.choices[0].text

        score_value = (
            1.0
            if self._compare_mcp_tool_calls(model_response, expected_mcp_call_dict)
            else 0.0
        )
        return score_value

    async def _extract_mcp_tool_call(self, model_response_text: str) -> Optional[Dict]:
        """
        Attempts to parse the model_response_text as a single JSON object.
        """
        try:
            # If the model includes <think> tags, strip them first if they are outside the JSON
            # For simplicity, assuming the model's primary output after <think> is the JSON
            if "</think>" in model_response_text:
                model_response_text = model_response_text.split("</think>", 1)[
                    -1
                ].strip()

            return json.loads(model_response_text)
        except json.JSONDecodeError:
            return None  # Failed to parse as JSON
        except Exception:  # Other potential errors
            return None

    async def _compare_mcp_tool_calls(
        self, model_response_text: str, expected_mcp_call_dict: Dict
    ) -> bool:
        """
        Compares the model's generated MCP tool call with the expected one.
        Returns:
            True if the model's output is a valid MCP call and matches the expected, False otherwise.
        """
        model_mcp_call = await self._extract_mcp_tool_call(model_response_text)

        if not model_mcp_call or not isinstance(model_mcp_call, dict):
            return False  # Model did not produce a valid JSON object

        if not isinstance(expected_mcp_call_dict, dict):
            # This shouldn't happen if your dataset is correctly formatted
            return False

        # Check tool_name
        if model_mcp_call.get("tool_name") != expected_mcp_call_dict.get("tool_name"):
            return False

        # Check arguments using a simplified version of your _json_objects_match or direct comparison
        # For robustness, you might want to ensure all expected arguments are present and match.
        model_args = model_mcp_call.get("arguments", {})
        expected_args = expected_mcp_call_dict.get("arguments", {})

        if not isinstance(model_args, dict) or not isinstance(expected_args, dict):
            return False  # Arguments are not dictionaries

        # A simple check: are expected_args a subset of model_args with matching values?
        for key, expected_value in expected_args.items():
            if key not in model_args or model_args[key] != expected_value:
                return False
        # Optionally, you could also penalize extra arguments in model_args if strictness is needed.

        return True  # All checks passed

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

    async def collect_trajectories(
        self, item
    ) -> Tuple[ScoredDataGroup, List]:  # this one
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
            max_tokens=1024,
            temperature=0.8,  # Using temperature to get diverse responses
        )

        to_score = list()
        for completion_choice in completions.choices:
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
        # if rollout_group_data is None: # Null check
        #     return scores # Return empty if nothing to process

        # if not rollout_group_data:
        #     print("rollout_group_data is empty, skipping")
        #     return None

        # NEW CODE THAT IS WORKING!!!!
        for item in rollout_group_data:
            model_response = item[0][-1]["content"]
            expected_mcp_call_dict = item[1]

            reward = (
                1.0
                if await self._compare_mcp_tool_calls(
                    model_response, expected_mcp_call_dict
                )
                else -1.0
            )

            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(reward)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        if all(scores["scores"][0] == score for score in scores["scores"]):
            return None

        return scores

    async def get_next_item(self):
        next_item_data = self.train[self.iter % len(self.train)]
        self.iter += 1

        user_prompt_content = next_item_data["user_prompt_text"]
        expected_mcp_call_dict = next_item_data["expected_mcp_call"]

        prompt_messages = []

        prompt_messages.append(
            frozenset({"role": "user", "content": user_prompt_content}.items())
        )

        answer = expected_mcp_call_dict  # This should be a Python dict

        return (tuple(prompt_messages), answer)

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
    McpEnv.cli()
