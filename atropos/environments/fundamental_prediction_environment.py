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

# System prompt only contains thinking instructions
system_prompt = """You are a deep thinking AI financial analyst.
You may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering.

You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your final prediction."""  # noqa E501

# User message template that contains task instructions
user_message_template = """Your task is to analyze the following company fundamentals, news, and macroeconomic data to predict whether the company's {fundamental_metric} will be maintained, raised, or reduced in the next quarter, as well as the magnitude of any change.

Your final answer MUST use the exact format:
"The {fundamental_metric} will be: {{answer}} and the magnitude will be: {{percentage}}%"

Where {{answer}} is one of: "maintained", "raised", or "reduced"
And {{percentage}} is the expected percentage change (0% if maintained).

Here is the data to analyze:

{context}"""  # noqa E501


class FundamentalPredictionEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        """
        Initialize the Fundamental Metric Prediction environment.

        Args:
            config: Configuration for the base environment
            server_configs: List of server configurations for OpenAI API
            slurm: Whether to use Slurm for distributed training
            testing: Whether in testing mode
        """
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.magnitude_accuracy_buffer = list()
        self.eval_metrics = list()

    @classmethod
    def config_init(self) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            max_num_workers=128,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="fundamental_metric_prediction",
            data_path_to_save_groups=None,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=256,
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """
        Set up the environment by loading and preparing the dataset.
        """
        # Load the full dataset
        full_dataset = load_dataset(
            "NousResearch/company-fundamentals-prediction-lite",
            "default",
            split="train",
        )

        full_dataset = full_dataset.shuffle(seed=42)

        # Create train/test split (95% train, 5% test)
        split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)

        # Keep the splits as is - no need to reformat
        self.train = split_dataset["train"]
        self.test = split_dataset["test"]

        # Print some dataset statistics
        print(
            f"Loaded dataset with {len(self.train)} training examples and {len(self.test)} test examples"
        )
        print(f"Example item format: {self.train[0]}")

        # Initialize iteration counter
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def get_next_item(self):
        """
        Get the next training item from the dataset.

        Returns:
            A tuple containing prompt and expected answer
        """
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1

        # Extract context, answer, magnitude and fundamental metric from the dataset item
        context = next_item["context"]
        answer = next_item["answer"]  # "maintained", "raised", or "reduced"
        magnitude = next_item["magnitude"]  # Percentage as string
        fundamental_metric = next_item[
            "fundamental_metric"
        ]  # Type of metric to predict

        # Create prompt tuple using frozensets as required
        prompt = []

        # Add system prompt
        prompt.append(frozenset({"role": "system", "content": system_prompt}.items()))

        # Format user message with context and fundamental metric
        user_content = user_message_template.format(
            context=context, fundamental_metric=fundamental_metric
        )
        prompt.append(frozenset({"role": "user", "content": user_content}.items()))

        return (tuple(prompt), answer, magnitude, fundamental_metric)

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        """
        Generate and collect model responses for scoring.

        Args:
            item: Input item containing prompt and expected answer

        Returns:
            Tuple of lists containing scored data groups and backlog
        """
        # Extract messages from the item
        messages = []
        for role_dict in item[0]:
            messages.append(dict(role_dict))

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get completions from the model
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=1024 * 15,
            temperature=0.8,  # Using higher temperature for diverse responses
        )

        to_score = list()

        for _, completion_choice in enumerate(completions.choices):
            # Create a copy of the prompt messages
            trajectory_messages = []
            for role_dict in item[0]:
                trajectory_messages.append(dict(role_dict))

            # Add the model's response
            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.text}
            )

            # Add to scoring queue with expected answer, magnitude, and fundamental metric
            to_score.append(
                (
                    tuple(trajectory_messages),
                    item[1],  # answer (maintained/raised/reduced)
                    item[2],  # magnitude
                    item[3],  # fundamental_metric
                )
            )

        # Call score to get the scored data
        scored_data = await self.score(to_score)
        to_backlog = []

        return scored_data, to_backlog

    def _extract_prediction(self, text, fundamental_metric):
        """
        Extract the prediction and magnitude from the model's response.

        Args:
            text: Text containing the model's response
            fundamental_metric: The fundamental metric being predicted

        Returns:
            Tuple of (prediction, magnitude) or (None, None) if extraction fails
        """
        # Check for thinking section
        think_tags = re.findall(r"<think>", text, re.IGNORECASE)
        think_close_tags = re.findall(r"</think>", text, re.IGNORECASE)

        # Verify thinking format - must have exactly one opening and one closing tag
        if len(think_tags) != 1 or len(think_close_tags) != 1:
            return None, None

        # Split on </think> to separate thinking from answer
        parts = re.split(r"</think>", text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) != 2:
            return None, None

        thinking_section, answer_section = parts

        # Validate thinking section contains opening tag
        if "<think>" not in thinking_section.lower():
            return None, None

        # Escape fundamental_metric for regex
        escaped_metric = re.escape(fundamental_metric)

        # Extract prediction and magnitude using regex - dynamic to match the fundamental metric
        pattern = f"The {escaped_metric} will be:\\s*(maintained|raised|reduced)\\s*and\\s*the\\s*magnitude\\s*will\\s*be:\\s*([-+]?\\d+(?:\\.\\d+)?)%"  # noqa E501

        # Find all matches to check if there are multiple predictions
        all_matches = re.findall(pattern, answer_section, re.IGNORECASE)

        # If no matches or multiple matches found, return None
        if len(all_matches) != 1:
            return None, None

        # Extract single match
        matches = re.search(pattern, answer_section, re.IGNORECASE)
        prediction = matches.group(1).lower()
        magnitude = matches.group(2)

        return prediction, magnitude

    def _calculate_magnitude_score(self, predicted_magnitude, expected_magnitude):
        """
        Calculate a score for magnitude prediction accuracy.

        Args:
            predicted_magnitude: The model's predicted magnitude percentage
            expected_magnitude: The expected magnitude percentage

        Returns:
            Score between 0.0 and 1.0 based on how close the prediction is
        """
        try:
            # Convert to float for comparison
            pred_mag = float(predicted_magnitude)
            exp_mag = float(expected_magnitude)

            # Calculate absolute difference
            diff = abs(pred_mag - exp_mag)

            # Score based on closeness to expected magnitude
            # Perfect match = 1.0
            # Within 1% = 0.9
            # Within 5% = 0.7
            # Within 10% = 0.5
            # Within 20% = 0.3
            # More than 20% off = 0.0

            if diff == 0:
                return 1.0
            elif diff <= 1:
                return 0.9
            elif diff <= 5:
                return 0.7
            elif diff <= 10:
                return 0.5
            elif diff <= 20:
                return 0.3
            else:
                return 0.0

        except ValueError:
            # If conversion fails, return 0
            return 0.0

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """
        Score the generated model responses for fundamental metric predictions.

        Args:
            rollout_group_data: List of generated responses with expected answers

        Returns:
            ScoredDataGroup with tokenized inputs and scores, or None if no valid scores
        """
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # Get the expected answer, magnitude, and fundamental metric
        expected_answer = rollout_group_data[0][
            1
        ]  # "maintained", "raised", or "reduced"
        expected_magnitude = rollout_group_data[0][2]  # Expected percentage change
        fundamental_metric = rollout_group_data[0][3]  # Type of fundamental metric

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            # Extract the model's response
            model_response = item[0][-1]["content"]

            # Extract the prediction and magnitude from the model's response
            prediction, magnitude = self._extract_prediction(
                model_response, fundamental_metric
            )

            # Calculate final score
            if prediction is None:
                final_score = 0.0  # Invalid format
            elif prediction == expected_answer:
                # Correct direction: base score of 1 + magnitude bonus
                magnitude_score = (
                    self._calculate_magnitude_score(magnitude, expected_magnitude)
                    if magnitude is not None
                    else 0.0
                )
                final_score = 1.0 + magnitude_score
            else:
                final_score = 0.0  # Incorrect direction

            # Apply length penalty for responses that are too long
            response_tokens = len(self.tokenizer.encode(model_response))
            if response_tokens > self.config.max_token_length * 0.95:
                # Penalize responses that are close to the max token limit
                final_score -= 0.5 * (response_tokens / self.config.max_token_length)

            # For binary reward signal, any positive score gets +1, otherwise -1
            binary_reward = 1.0 if final_score > 0 else -1.0

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(binary_reward)

            # For tracking metrics
            directional_correct = (
                1.0 if prediction == expected_answer and prediction is not None else 0.0
            )
            self.percent_correct_buffer.append(directional_correct)
            if prediction == expected_answer and magnitude is not None:
                self.magnitude_accuracy_buffer.append(
                    self._calculate_magnitude_score(magnitude, expected_magnitude)
                )

            # Break once we have enough examples
            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Return None if all scores are the same (no learning signal)
        if all(scores["scores"][0] == score for score in scores["scores"]):
            return None

        return scores

    async def rollout_and_score_eval(self, test_item):
        """
        Generate and score model responses for a single test item.

        Args:
            test_item: Test item from dataset

        Returns:
            Dictionary with direction and magnitude scores
        """
        # Extract context, answer, magnitude and fundamental metric from the test item
        context = test_item["context"]
        expected_answer = test_item["answer"]
        expected_magnitude = test_item["magnitude"]
        fundamental_metric = test_item["fundamental_metric"]

        # Format user message with context and fundamental metric
        user_content = user_message_template.format(
            context=context, fundamental_metric=fundamental_metric
        )

        # Create messages for model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get model completion
        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=1024 * 16,
            temperature=0.2,  # Lower for eval
            split="eval",
        )

        # Extract the model's response
        model_response = completion.choices[0].text

        # Extract prediction and magnitude
        prediction, magnitude = self._extract_prediction(
            model_response, fundamental_metric
        )

        # Calculate direction score (1 for correct, 0 for incorrect)
        direction_score = (
            1 if prediction == expected_answer and prediction is not None else 0
        )

        # Calculate magnitude score if direction is correct
        magnitude_score = 0
        if direction_score == 1 and magnitude is not None:
            magnitude_score = self._calculate_magnitude_score(
                magnitude, expected_magnitude
            )

        # Calculate combined score (1 + magnitude_score for correct direction, 0 for incorrect)
        combined_score = (1 + magnitude_score) if direction_score == 1 else 0

        return {
            "direction_score": direction_score,
            "magnitude_score": magnitude_score,
            "combined_score": combined_score,
        }

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the model on test data.
        """
        eval_tasks = []
        for test_item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(test_item))

        # Run evaluation
        all_scores = await tqdm_asyncio.gather(*eval_tasks)

        # Calculate aggregate metrics
        direction_scores = [score["direction_score"] for score in all_scores]
        magnitude_scores = [
            score["magnitude_score"]
            for score in all_scores
            if score["direction_score"] == 1
        ]
        combined_scores = [score["combined_score"] for score in all_scores]

        # Calculate and log metrics
        direction_accuracy = (
            sum(direction_scores) / len(direction_scores) if direction_scores else 0
        )
        magnitude_accuracy = (
            sum(magnitude_scores) / len(magnitude_scores) if magnitude_scores else 0
        )
        average_combined_score = (
            sum(combined_scores) / len(combined_scores) if combined_scores else 0
        )

        self.eval_metrics.append(("eval/direction_accuracy", direction_accuracy))
        self.eval_metrics.append(("eval/magnitude_accuracy", magnitude_accuracy))
        self.eval_metrics.append(("eval/combined_score", average_combined_score))

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Calculate and log training direction accuracy
        try:
            direction_accuracy = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
            wandb_metrics["train/direction_accuracy"] = direction_accuracy
        except ZeroDivisionError:
            pass  # Skip if buffer is empty

        # Calculate and log training magnitude accuracy
        try:
            magnitude_accuracy = sum(self.magnitude_accuracy_buffer) / len(
                self.magnitude_accuracy_buffer
            )
            wandb_metrics["train/magnitude_accuracy"] = magnitude_accuracy
        except ZeroDivisionError:
            pass  # Skip if buffer is empty

        # Calculate combined training score (direction + magnitude)
        try:
            combined_score = (
                direction_accuracy + magnitude_accuracy
                if "direction_accuracy" in wandb_metrics
                else 0
            )
            wandb_metrics["train/combined_score"] = combined_score
        except Exception as e:
            print(f"Error calculating combined score: {e}")
            pass

        # Clear the buffers after logging
        self.percent_correct_buffer = list()
        self.magnitude_accuracy_buffer = list()

        # Log evaluation metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        # Initialize rollouts_for_wandb if not exists
        if not hasattr(self, "rollouts_for_wandb"):
            self.rollouts_for_wandb = []

        # Get number of examples to keep
        num_keep = getattr(self.config, "num_rollouts_per_group_for_logging", -1)

        if num_keep == -1:
            num_keep = self.config.group_size

        # Get fundamental metric from item
        fundamental_metric = item[3]

        # Add examples to rollouts
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored_data["tokens"][i]),
                    scored_data["scores"][i],
                    item[1],  # expected direction (maintained/raised/reduced)
                    item[2],  # expected magnitude
                    fundamental_metric,  # metric type being predicted
                )
                for i in range(min(num_keep, len(scored_data["tokens"])))
            ]
        )

        # Keep buffer size limited
        max_rollouts = getattr(self.config, "num_rollouts_to_keep", 10)
        if len(self.rollouts_for_wandb) > max_rollouts:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics):
        if hasattr(self, "rollouts_for_wandb") and len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(
                columns=[
                    "text",
                    "score",
                    "expected_direction",
                    "expected_magnitude",
                    "fundamental_metric",
                ]
            )

            for group in self.rollouts_for_wandb:
                for item in group:
                    table.add_data(item[0], item[1], item[2], item[3], item[4])

            wandb_metrics["train/rollouts"] = table

        # Clear rollouts after logging
        self.rollouts_for_wandb = []

        return wandb_metrics


if __name__ == "__main__":
    FundamentalPredictionEnv.cli()
