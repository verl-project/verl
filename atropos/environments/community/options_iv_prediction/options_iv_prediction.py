import random
import re
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import wandb
from datasets import Dataset
from tqdm.asyncio import tqdm_asyncio
from yahooquery import Ticker

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
system_prompt = """You are a deep thinking AI Stock Options analyst.
You may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering.

You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your final prediction."""  # noqa E501

# User message template that contains task instructions
user_message_template = (
    "Your task is to analyze the following option data and predict the implied volatility of the option.\n\n"
    "Option Price: ${option_price:.2f}\n"
    "Stock Price: ${stock_price:.2f}\n"
    "Strike Price: ${strike_price:.2f}\n"
    "Time to Expiry: {time_to_expiry:.6f} years\n"
    "Risk-Free Rate: {risk_free_rate:.2f} \n\n"
    'Your final answer MUST use the exact format: "The implied volatility will be: {{answer}}"\n'
    "Where {{answer}} is the implied volatility as a string in percent (e.g. 70%)"
)


class OptionsIVPrediction(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        """
        Initialize the Options Implied Volatility Prediction environment.

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
            wandb_name="options_iv_prediction",
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
        # Use yahooquery to get option data
        stocks = ["UNH"]
        unh = Ticker(stocks)
        df = unh.option_chain
        stock_price = unh.financial_data["UNH"]["currentPrice"]
        risk_free_rate = 0.05  # Fixed risk-free rate

        # Process the options data
        processed_data = []
        for index, row in df.iterrows():
            try:
                option_price = row["lastPrice"]
                strike_price = row["strike"]
                expiry = pd.Timestamp(index[1])  # expiry date
                now = pd.Timestamp.now()
                time_to_expiration = (expiry - now).total_seconds() / (
                    365.25 * 24 * 60 * 60
                )

                # Skip invalid options
                if option_price <= 0 or time_to_expiration <= 0:
                    continue

                # Get the implied volatility directly from the row
                iv = row["impliedVolatility"]

                # Format as a percentage
                iv_percentage = f"{iv * 100:.2f}%"

                # Create context dictionary
                context = {
                    "option_price": option_price,
                    "strike_price": strike_price,
                    "time_to_expiry": time_to_expiration,
                    "risk_free_rate": risk_free_rate,
                    "stock_price": stock_price,
                }

                processed_data.append(
                    {
                        "context": context,
                        "answer": iv_percentage,
                        "raw_iv": iv * 100,  # Store raw value for scoring
                    }
                )
            except Exception as e:
                # Skip any options that cause errors
                print(row["expiration"])
                print(f"Skipping option due to error: {e}")
                continue

        # Convert to dataset
        dataset = Dataset.from_dict(
            {
                "context": [item["context"] for item in processed_data],
                "answer": [item["answer"] for item in processed_data],
                "raw_iv": [item["raw_iv"] for item in processed_data],
            }
        )

        # Create train/test split (95% train, 5% test)
        split_dataset = dataset.shuffle(seed=42).train_test_split(
            test_size=0.05, seed=42
        )

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

        # Extract context and answer from the dataset item
        context = next_item["context"]
        answer = next_item["answer"]  # IV as percentage string
        raw_iv = next_item["raw_iv"]  # Raw IV as float

        # Create prompt tuple using frozensets as required
        prompt = []

        # Add system prompt
        prompt.append(frozenset({"role": "system", "content": system_prompt}.items()))

        # Format user message with context
        print(context)
        user_content = user_message_template.format(**context)
        prompt.append(frozenset({"role": "user", "content": user_content}.items()))

        return (tuple(prompt), answer, raw_iv)

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
            max_tokens=1024,
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

            # Add to scoring queue with expected answer and raw IV
            to_score.append(
                (
                    tuple(trajectory_messages),
                    item[1],  # answer (formatted IV percentage)
                    item[2],  # raw_iv (floating point value)
                )
            )

        # Call score to get the scored data
        scored_data = await self.score(to_score)
        to_backlog = []

        return scored_data, to_backlog

    def _extract_prediction(self, text):
        """
        Extract the implied volatility prediction from the model's response.

        Args:
            text: Text containing the model's response

        Returns:
            The extracted IV as a string or None if extraction fails
        """
        # Check for thinking section
        think_tags = re.findall(r"<think>", text, re.IGNORECASE)
        think_close_tags = re.findall(r"</think>", text, re.IGNORECASE)

        # Verify thinking format - must have exactly one opening and one closing tag
        if len(think_tags) != 1 or len(think_close_tags) != 1:
            return None

        # Split on </think> to separate thinking from answer
        parts = re.split(r"</think>", text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) != 2:
            return None

        thinking_section, answer_section = parts

        # Validate thinking section contains opening tag
        if "<think>" not in thinking_section.lower():
            return None

        # Extract IV prediction using regex
        pattern = r"The implied volatility will be:\s*([\d.]+%)"

        # Find all matches to check if there are multiple predictions
        all_matches = re.findall(pattern, answer_section, re.IGNORECASE)

        # If no matches or multiple matches found, return None
        if len(all_matches) != 1:
            return None

        # Extract single match
        matches = re.search(pattern, answer_section, re.IGNORECASE)
        prediction = matches.group(1)

        return prediction

    def _calculate_iv_score(self, predicted_iv, expected_iv):
        """
        Calculate a score for IV prediction accuracy.

        Args:
            predicted_iv: The model's predicted IV percentage
            expected_iv: The expected IV percentage as a float

        Returns:
            Score between 0.0 and 1.0 based on how close the prediction is
        """
        try:
            # Convert predicted percentage to float
            if isinstance(predicted_iv, str) and "%" in predicted_iv:
                pred_iv = float(predicted_iv.strip("%"))
            else:
                pred_iv = float(predicted_iv)

            # Expected IV is already a float
            exp_iv = float(expected_iv)

            # Calculate absolute difference
            diff = abs(pred_iv - exp_iv)

            # Score based on closeness to expected IV
            # Perfect match = 1.0
            # Within 1% = 0.9
            # Within 5% = 0.7
            # Within 10% = 0.5
            # Within 20% = 0.3
            # More than 20% off = 0.0

            if diff < 0.5:
                return 1.0
            elif diff <= 2:
                return 0.95
            elif diff <= 5:
                return 0.85
            elif diff <= 10:
                return 0.7
            elif diff <= 15:
                return 0.5
            elif diff <= 20:
                return 0.3
            else:
                return 0.1

        except ValueError:
            # If conversion fails, return 0
            return 0.0

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """
        Score the generated model responses for IV predictions.

        Args:
            rollout_group_data: List of generated responses with expected answers

        Returns:
            ScoredDataGroup with tokenized inputs and scores, or None if no valid scores
        """
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # Get the expected raw IV
        expected_raw_iv = rollout_group_data[0][2]  # Raw IV as float

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            # Extract the model's response
            model_response = item[0][-1]["content"]

            # Extract the prediction from the model's response
            prediction = self._extract_prediction(model_response)

            # Calculate final score
            if prediction is None:
                final_score = -0.5  # Invalid format
            else:
                # Calculate IV accuracy score
                iv_score = self._calculate_iv_score(prediction, expected_raw_iv)
                final_score = iv_score

            # Apply length penalty for responses that are too long
            response_tokens = len(self.tokenizer.encode(model_response))
            if response_tokens > self.config.max_token_length * 0.95:
                # Penalize responses that are close to the max token limit
                final_score -= 0.5 * (response_tokens / self.config.max_token_length)

            # For binary reward signal, any positive score gets +1, otherwise -1
            binary_reward = 1.0 if final_score > 0.5 else -1.0

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
            if prediction is not None:
                try:
                    pred_iv = float(prediction.strip("%"))
                    accuracy = self._calculate_iv_score(pred_iv, expected_raw_iv)
                    self.percent_correct_buffer.append(1.0 if accuracy >= 0.7 else 0.0)
                    self.magnitude_accuracy_buffer.append(accuracy)
                except ValueError:
                    self.percent_correct_buffer.append(0.0)
                    self.magnitude_accuracy_buffer.append(0.0)
            else:
                self.percent_correct_buffer.append(0.0)
                self.magnitude_accuracy_buffer.append(0.0)

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
            Dictionary with IV prediction accuracy
        """
        # Extract context and answer from the test item
        context = test_item["context"]
        expected_answer = test_item["answer"]
        expected_raw_iv = test_item["raw_iv"]

        # Format user message with context
        user_content = user_message_template.format(context=context)

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

        # Extract prediction
        prediction = self._extract_prediction(model_response)

        # Calculate scores
        format_score = 1 if prediction is not None else 0

        accuracy_score = 0
        if prediction is not None:
            try:
                pred_iv = float(prediction.strip("%"))
                accuracy_score = self._calculate_iv_score(pred_iv, expected_raw_iv)
            except ValueError:
                accuracy_score = 0

        # Binary score - correct if within 10% of actual IV
        binary_score = 1 if accuracy_score >= 0.7 else 0

        return {
            "format_score": format_score,
            "accuracy_score": accuracy_score,
            "binary_score": binary_score,
            "predicted_iv": prediction if prediction is not None else "invalid",
            "expected_iv": expected_answer,
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
        format_scores = [score["format_score"] for score in all_scores]
        accuracy_scores = [
            score["accuracy_score"]
            for score in all_scores
            if score["format_score"] == 1
        ]
        binary_scores = [score["binary_score"] for score in all_scores]

        # Calculate and log metrics
        format_accuracy = (
            sum(format_scores) / len(format_scores) if format_scores else 0
        )
        iv_accuracy = (
            sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        )
        binary_accuracy = (
            sum(binary_scores) / len(binary_scores) if binary_scores else 0
        )

        self.eval_metrics.append(("eval/format_accuracy", format_accuracy))
        self.eval_metrics.append(("eval/iv_accuracy", iv_accuracy))
        self.eval_metrics.append(("eval/binary_accuracy", binary_accuracy))

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Calculate and log training format accuracy
        try:
            format_accuracy = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
            wandb_metrics["train/format_accuracy"] = format_accuracy
        except ZeroDivisionError:
            pass  # Skip if buffer is empty

        # Calculate and log training IV accuracy
        try:
            iv_accuracy = sum(self.magnitude_accuracy_buffer) / len(
                self.magnitude_accuracy_buffer
            )
            wandb_metrics["train/iv_accuracy"] = iv_accuracy
        except ZeroDivisionError:
            pass  # Skip if buffer is empty

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

        # Add examples to rollouts
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored_data["tokens"][i]),
                    scored_data["scores"][i],
                    item[1],  # expected IV as string
                    item[2],  # expected IV as raw value
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
                    "expected_iv_string",
                    "expected_iv_raw",
                ]
            )

            for group in self.rollouts_for_wandb:
                for item in group:
                    table.add_data(item[0], item[1], item[2], item[3])

            wandb_metrics["train/rollouts"] = table

        # Clear rollouts after logging
        self.rollouts_for_wandb = []

        return wandb_metrics


if __name__ == "__main__":
    OptionsIVPrediction.cli()
